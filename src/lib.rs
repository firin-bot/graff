#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use anyhow::anyhow;
use anyhow::Context as _;
use anyhow::Result;
use core::cell::RefCell;
use derive_more::Deref;
use derive_more::DerefMut;
use derive_more::IntoIterator;
use hashbrown::HashMap;
use heapless::format;
use heapless::String;
use petgraph::acyclic::Acyclic;
use petgraph::data::Build as _;
use petgraph::data::DataMapMut as _;
use petgraph::Direction;
use petgraph::stable_graph::NodeIndex;
use petgraph::stable_graph::StableDiGraph;
use petgraph::visit::EdgeRef as _;
use petgraph::visit::IntoEdgeReferences as _;

#[derive(Debug)]
pub enum Kind {
    Type,
    Arrow(Box<Self>, Box<Self>)
}

impl Kind {
    pub fn arrow(a: Self, b: Self) -> Self {
        Self::Arrow(Box::new(a), Box::new(b))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypeCon {
    // arity = 0
    Boolean,
    Character,
    Integer,
    Real,

    // arity = 1
    Effect,
    List,

    // arity = 2
    Arrow,

    // arity = N
    Tuple(usize)
}

impl TypeCon {
    pub fn kind(&self) -> Kind {
        match self {
            Self::Boolean   => Kind::Type,
            Self::Character => Kind::Type,
            Self::Integer   => Kind::Type,
            Self::Real      => Kind::Type,

            Self::Effect    => Kind::arrow(Kind::Type, Kind::Type),
            Self::List      => Kind::arrow(Kind::Type, Kind::Type),

            Self::Arrow     => Kind::arrow(Kind::Type, Kind::arrow(Kind::Type, Kind::Type)),

            Self::Tuple(n)  => (0..*n).fold(Kind::Type, |acc, _| Kind::arrow(Kind::Type, acc))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TypeVar(pub u32);

#[derive(Clone, Debug)]
pub enum Type {
    Var(TypeVar),
    App(TypeCon, Vec<Type>)
}

#[derive(Clone, Default, Deref, DerefMut, IntoIterator)]
pub struct SubstitutionMap(HashMap<TypeVar, Type>);

impl SubstitutionMap {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    pub fn compose_with(&mut self, s: &Self) {
        for ty in self.values_mut() {
            *ty = ty.substitute(s);
        }
        self.extend(s.clone());
    }
}

impl core::iter::FromIterator<(TypeVar, Type)> for SubstitutionMap {
    fn from_iter<I: IntoIterator<Item = (TypeVar, Type)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl Type {
    pub const fn boolean() -> Self {
        Self::App(TypeCon::Boolean, vec![])
    }

    pub const fn character() -> Self {
        Self::App(TypeCon::Character, vec![])
    }

    pub const fn integer() -> Self {
        Self::App(TypeCon::Integer, vec![])
    }

    pub const fn real() -> Self {
        Self::App(TypeCon::Real, vec![])
    }

    pub fn effect(inner_ty: Self) -> Self {
        Self::App(TypeCon::Effect, vec![inner_ty])
    }

    pub fn list(elem_ty: Self) -> Self {
        Self::App(TypeCon::List, vec![elem_ty])
    }

    pub fn arrow(a: Self, b: Self) -> Self {
        Self::App(TypeCon::Arrow, vec![a, b])
    }

    pub const fn tuple(elem_tys: Vec<Self>) -> Self {
        Self::App(TypeCon::Tuple(elem_tys.len()), elem_tys)
    }

    pub const fn unit() -> Self {
        Self::tuple(vec![])
    }

    pub fn singleton(elem_ty: Self) -> Self {
        Self::tuple(vec![elem_ty])
    }

    pub fn break_arrow(&self) -> Option<(&Self, &Self)> {
        if let Self::App(TypeCon::Arrow, args) = self {
            Some((args.first()?, args.get(1)?))
        } else {
            None
        }
    }

    pub fn break_tuple(&self) -> Option<&[Self]> {
        if let Self::App(TypeCon::Tuple(_), args) = self {
            Some(args)
        } else {
            None
        }
    }

    pub fn substitute(&self, subs: &SubstitutionMap) -> Self {
        match self {
            Self::Var(v) => subs.get(v).unwrap_or(self).clone(),
            Self::App(con, args) => Self::App(
                con.clone(),
                args.iter().map(|a| a.substitute(subs)).collect()
            )
        }
    }

    pub fn unify(t0: &Self, t1: &Self) -> Result<SubstitutionMap> {
        fn bind(a: TypeVar, t: &Type) -> Result<SubstitutionMap> {
            Ok(core::iter::once((a, t.clone())).collect())
        }

        match (t0, t1) {
            (Self::Var(a), Self::Var(b)) if a == b => Ok(SubstitutionMap::default()),
            (Self::Var(a), t) => bind(*a, t),
            (t, Self::Var(a)) => bind(*a, t),
            (Self::App(con0, a0), Self::App(con1, a1)) if con0 == con1 && a0.len() == a1.len() => {
                let mut subs = SubstitutionMap::default();
                for (x, y) in a0.iter().zip(a1) {
                    let x = x.substitute(&subs);
                    let y = y.substitute(&subs);
                    subs.compose_with(&Self::unify(&x, &y)?);
                }
                Ok(subs)
            },
            _ => anyhow::bail!("type mismatch between {:?} and {:?}", t0, t1)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Pred {
    Num(TypeVar)
}

#[derive(Clone, Debug)]
pub struct Qual {
    pub preds: Vec<Pred>,
    pub ty: Type
}

impl Qual {
    pub fn substitute(&self, subs: &SubstitutionMap) -> Self {
        Self {
            preds: self.preds.clone(),
            ty: self.ty.substitute(subs)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scheme {
    pub vars: Vec<TypeVar>,
    pub qual: Qual
}

impl Scheme {
    pub fn instantiate(&self, ctx: &mut Context) -> Qual {
        let mut subs = SubstitutionMap::with_capacity(self.vars.len());
        for v in &self.vars {
            subs.insert(*v, Type::Var(ctx.fresh_var()));
        }
        self.qual.substitute(&subs)
    }
}

#[derive(Clone)]
pub struct Effect {
    ret_ty: Type,
    thunk: Rc<dyn Fn() -> Result<Value>>
}

impl Effect {
    pub const fn ret_ty(&self) -> &Type {
        &self.ret_ty
    }

    pub fn run(&self) -> Result<Value> {
        (self.thunk)()
    }
}

impl core::fmt::Debug for Effect {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str("<effect>")
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Boolean(bool),
    Character(char),
    Integer(i64),
    Real(f64),
    Tuple(Vec<Value>),
    List {
        elem_ty: Type,
        elems: Vec<Value>
    },
    Effect(Effect),
    Instance(Box<Instance>)
}

impl Value {
    pub const fn unit() -> Self {
        Self::Tuple(vec![])
    }

    pub fn singleton(elem_ty: Self) -> Self {
        Self::Tuple(vec![elem_ty])
    }

    pub fn ty(&self) -> Type {
        match self {
            Self::Boolean(_)           => Type::boolean(),
            Self::Character(_)         => Type::character(),
            Self::Integer(_)           => Type::integer(),
            Self::Real(_)              => Type::real(),
            Self::Tuple(vals)          => Type::tuple(vals.iter().map(Self::ty).collect()),
            Self::List { elem_ty, .. } => Type::list(elem_ty.clone()),
            Self::Effect(effect)       => Type::effect(effect.ret_ty.clone()),
            Self::Instance(instance)   => instance.qual.ty.clone()
        }
    }

    pub fn require_integer(&self) -> Result<i64> {
        if let Self::Integer(x) = self { Ok(*x) } else { Err(anyhow!("Value not an integer")) }
    }
}

pub type Symbol = alloc::string::String;

#[derive(Clone, Debug)]
pub struct Prototype {
    pub symbol: Symbol,
    pub scheme: Scheme
}

#[derive(Clone, Debug)]
pub enum Op {
    Constant(Value),
    Pure,
    Bind,
    Graph(Box<Graph>),
    Binding(Prototype),
    Add,
}

impl From<Graph> for Op {
    fn from(graph: Graph) -> Self {
        Self::Graph(Box::new(graph))
    }
}

impl From<Prototype> for Op {
    fn from(prototype: Prototype) -> Self {
        Self::Binding(prototype)
    }
}

impl Op {
    pub fn scheme(&self) -> Scheme {
        match self {
            Self::Constant(v) => scheme!(() -> ({ v.ty() })),
            Self::Pure => scheme!(forall a. (a) -> (Effect a)),
            Self::Bind => scheme!(forall a b. (Effect a, (a) -> (Effect b)) -> (Effect b)),
            Self::Graph(g) => g.scheme.clone(),
            Self::Binding(prototype) => prototype.scheme.clone(),
            Self::Add => scheme!(forall a. Num a => (a, a) -> (a)),
        }
    }

    pub fn instantiate(&self, ctx: &mut Context) -> Instance {
        Instance {
            data: self.clone().into(),
            qual: self.scheme().instantiate(ctx)
        }
    }
}

#[derive(Clone, Debug)]
pub enum InstanceData {
    Op(Op),
    Input(usize),
    Output(usize)
}

impl From<Op> for InstanceData {
    fn from(op: Op) -> Self {
        Self::Op(op)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FromPort {
    Instance,
    Index(usize)
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ToPort(usize);

#[derive(Clone, Debug)]
pub struct Instance {
    pub data: InstanceData,
    pub qual: Qual
}

#[derive(Clone, Default)]
pub struct Inputs {
    elements: [Option<Value>; 8]
}

impl Inputs {
    pub fn insert(&mut self, index: usize, v: Value) -> Result<()> {
        let message1: String<28> = format!("input index {} out of bounds", index)?;
        *self.elements.get_mut(index).context(message1)? = Some(v);
        Ok(())
    }

    pub fn require(&self, index: usize) -> Result<&Value> {
        let message1: String<28> = format!("input index {} out of bounds", index)?;
        let message2: String<28> = format!("input index {} not connected", index)?;
        self.elements.get(index).context(message1)?.as_ref().context(message2)
    }
}

impl Instance {
    pub fn type_of_from_port(&self, port: FromPort) -> Result<&Type> {
        match port {
            FromPort::Instance => Ok(&self.qual.ty),
            FromPort::Index(index) => {
                let (_, o) = self.qual.ty.break_arrow().context("failed to break type as Arrow")?;
                let tuple = o.break_tuple().context("failed to break Arrow output as Tuple")?;
                tuple.get(index).context("output port index out of bounds")
            }
        }
    }

    pub fn type_of_to_port(&self, ToPort(index): ToPort) -> Result<&Type> {
        let (i, _) = self.qual.ty.break_arrow().context("failed to break type as Arrow")?;
        let tuple = i.break_tuple().context("failed to break Arrow output as Tuple")?;
        tuple.get(index).context("input port index out of bounds")
    }

    pub fn evaluate(&self, library: RefCell<Library>, graph_inputs: &Inputs, inputs: &Inputs, from: usize) -> Result<Value> {
        let x = match &self.data {
            InstanceData::Op(op) => match op {
                Op::Constant(val) => Ok(val.clone()),
                Op::Pure          => {
                    let v = inputs.require(0)?.clone();
                    Ok(Value::Effect(Effect {
                        ret_ty: v.ty(),
                        thunk: Rc::new(move || Ok(v.clone()))
                    }))
                },
                Op::Bind          => {
                    let v0 = inputs.require(0)?.clone();
                    let v1 = inputs.require(1)?.clone();
                    match v0 {
                        Value::Effect(effect) => {
                            match v1 {
                                Value::Instance(instance) => {
                                    let graph_inputs = graph_inputs.clone();
                                    Ok(Value::Effect(Effect {
                                        ret_ty: instance.type_of_from_port(FromPort::Index(0))?.clone(),
                                        thunk: Rc::new(move || {
                                            let mut inputs = Inputs::default();
                                            inputs.insert(0, effect.run()?)?;
                                            match instance.evaluate(RefCell::clone(&library), &graph_inputs, &inputs, 0)? {
                                                Value::Effect(new_effect) => new_effect.run(),
                                                _ => Err(anyhow!("expected result of Bind instance to be an effect"))
                                            }
                                        })
                                    }))
                                }
                                _ => Err(anyhow!("expected v1 for Bind instance to be an instance"))
                            }
                        },
                        _ => Err(anyhow!("expected v0 for Bind instance to be an effect"))
                    }
                },
                Op::Graph(g) => g.evaluate(inputs, from).context("failed to evaluate sub-graph"),
                Op::Binding(prototype) => {
                    let owned_library = library.borrow();
                    match owned_library.require(prototype)? {
                        Binding::Graph(g)           => g.evaluate(inputs, from).context("failed to evaluate external sub-graph"),
                        Binding::External(external) => (external.f)(inputs, from)
                    }
                },
                Op::Add => {
                    let v0 = inputs.require(0)?;
                    let v1 = inputs.require(1)?;
                    match (v0, v1) {
                        (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x + y)),
                        (Value::Real(x),    Value::Real(y))    => Ok(Value::Real(x + y)),
                        _ => anyhow::bail!("invalid type for Add instance")
                    }
                }
            },
            InstanceData::Input(i) => graph_inputs.require(*i).context("graph input missing").cloned(),
            InstanceData::Output(_) => inputs.require(0).cloned()
        }?;
        Ok(x)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Context {
    next: u32
}

impl Context {
    pub const fn with_initial(initial: u32) -> Self {
        Self {
            next: initial
        }
    }

    pub const fn fresh_var(&mut self) -> TypeVar {
        let v = self.next;
        self.next += 1;
        TypeVar(v)
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    from: FromPort,
    to: ToPort
}

#[derive(Clone, Debug)]
pub struct Graph {
    ctx: Context,
    g: Acyclic<StableDiGraph<Instance, Edge>>,
    scheme: Scheme,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
    library: RefCell<Library>
}

impl Graph {
    pub fn new(scheme: Scheme) -> Result<Self> {
        let ctx = Context::with_initial(scheme.vars.len() as u32);
        let mut g: Acyclic<StableDiGraph<Instance, Edge>> = Default::default();

        let (in_ty, out_ty) = scheme.qual.ty.break_arrow().context("failed to break graph scheme as Arrow")?;
        let in_tuple = in_ty.break_tuple().context("failed to break graph input as Tuple")?;
        let out_tuple = out_ty.break_tuple().context("failed to break graph output as Tuple")?;

        let inputs: Vec<_> = in_tuple.iter().enumerate().map(|(i, ty)| {
            g.add_node(Instance {
                data: InstanceData::Input(i),
                qual: Qual {
                    preds: vec![],
                    ty: Type::arrow(
                        Type::unit(),
                        Type::singleton(ty.clone())
                    )
                }
            })
        }).collect();
        let outputs: Vec<_> = out_tuple.iter().enumerate().map(|(i, ty)| {
            g.add_node(Instance {
                data: InstanceData::Output(i),
                qual: Qual {
                    preds: vec![],
                    ty: Type::arrow(
                        Type::singleton(ty.clone()),
                        Type::unit()
                    )
                }
            })
        }).collect();

        Ok(Self {
            ctx,
            g,
            scheme,
            inputs,
            outputs,
            library: Default::default()
        })
    }

    pub const fn inner(&self) -> &Acyclic<StableDiGraph<Instance, Edge>> {
        &self.g
    }

    pub fn get_input(&self, index: usize) -> Result<NodeIndex> {
        let message: String<24> = format!("input index {} not found", index)?;
        self.inputs.get(index).copied().context(message)
    }

    pub fn get_output(&self, index: usize) -> Result<NodeIndex> {
        let message: String<25> = format!("output index {} not found", index)?;
        self.outputs.get(index).copied().context(message)
    }

    pub fn add(&mut self, op: Op) -> NodeIndex {
        self.g.add_node(op.instantiate(&mut self.ctx))
    }

    pub fn connect(&mut self, u: NodeIndex, from: usize, v: NodeIndex, to: usize) {
        self.g.add_edge(u, v, Edge {
            from: FromPort::Index(from),
            to: ToPort(to)
        });
    }

    pub fn connect_lambda(&mut self, u: NodeIndex, v: NodeIndex, to: usize) {
        self.g.add_edge(u, v, Edge {
            from: FromPort::Instance,
            to: ToPort(to)
        });
    }

    pub fn type_check(&mut self) -> Result<()> {
        let mut subs = SubstitutionMap::default();

        for e in self.g.edge_references() {
            let u = e.source();
            let v = e.target();
            let edge = e.weight();

            let t0 = self.g[u].type_of_from_port(edge.from)?.substitute(&subs);
            let t1 = self.g[v].type_of_to_port(edge.to)?.substitute(&subs);

            subs.compose_with(&Type::unify(&t0, &t1)?);
        }

        let indices: Vec<_> = self.g.nodes_iter().collect();
        for i in indices {
            if let Some(n) = self.g.node_weight_mut(i) {
                n.qual = n.qual.substitute(&subs);
            }
        }

        Ok(())
    }

    pub fn link(&mut self, lib: &Library) {
        self.library.borrow_mut().link(lib);
    }

    pub fn evaluate_index(&self, graph_inputs: &Inputs, index: NodeIndex, port: FromPort) -> Result<Value> {
        match port {
            FromPort::Instance => Ok(Value::Instance(Box::new(self.g[index].clone()))),
            FromPort::Index(from) => {
                let mut inputs = Inputs::default();
                for edge in self.g.edges_directed(index, Direction::Incoming) {
                    let v = self.evaluate_index(graph_inputs, edge.source(), edge.weight().from)?;
                    inputs.insert(edge.weight().to.0, v)?;
                }

                self.g[index].evaluate(RefCell::clone(&self.library), graph_inputs, &inputs, from)
            }
        }
    }

    pub fn evaluate(&self, graph_inputs: &Inputs, index: usize) -> Result<Value> {
        self.evaluate_index(graph_inputs, self.get_output(index).context("output port out of bounds")?, FromPort::Index(0))
    }
}

#[derive(Clone)]
pub struct External {
    scheme: Scheme,

    #[allow(clippy::type_complexity)]
    f: Rc<dyn Fn(&Inputs, usize) -> Result<Value>>
}

impl core::fmt::Debug for External {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str("<external code>")
    }
}

impl External {
    fn new<F: Fn(&Inputs, usize) -> Result<Value> + 'static>(
        scheme: Scheme,
        f: F
    ) -> Self {
        Self {
            scheme,
            f: Rc::new(f)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Binding {
    Graph(Box<Graph>),
    External(External)
}

impl Binding {
    pub const fn scheme(&self) -> &Scheme {
        match self {
            Self::Graph(graph)       => &graph.scheme,
            Self::External(external) => &external.scheme
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Library {
    bindings: HashMap<Symbol, Binding>
}

impl Library {
    pub fn insert_graph(&mut self, symbol: &str, graph: Graph) {
        self.bindings.insert(symbol.into(), Binding::Graph(Box::new(graph)));
    }

    pub fn insert_external<F: Fn(&Inputs, usize) -> Result<Value> + 'static>(
        &mut self,
        symbol: &str,
        scheme: Scheme,
        f: F
    ) {
        self.bindings.insert(symbol.into(), Binding::External(External::new(scheme, f)));
    }

    pub fn prototype(&self, symbol: &str) -> Result<Prototype> {
        let message: String<85> = format!("no binding found for `{}`", symbol)?;
        let binding = self.bindings.get(symbol).context(message)?;
        Ok(Prototype {
            symbol: symbol.into(),
            scheme: binding.scheme().clone()
        })
    }

    pub fn require(&self, prototype: &Prototype) -> Result<&Binding> {
        let message: String<85> = format!("no binding found for `{}`", prototype.symbol)?;
        self.bindings.get(&prototype.symbol).context(message)
    }

    pub fn link(&mut self, lib: &Self) {
        self.bindings.extend(lib.bindings.clone());
    }
}

#[macro_export]
macro_rules! typevar_index {
    (a) => { 0 };
    (b) => { 1 };
    (c) => { 2 };
    (d) => { 3 };
    (e) => { 4 };
    (f) => { 5 };
    (g) => { 6 };
    (h) => { 7 };
    (i) => { 8 };
    (j) => { 9 };
    (k) => { 10 };
    (l) => { 11 };
    (m) => { 12 };
    (n) => { 13 };
    (o) => { 14 };
    (p) => { 15 };
    (q) => { 16 };
    (r) => { 17 };
    (s) => { 18 };
    (t) => { 19 };
    (u) => { 20 };
    (v) => { 21 };
    (w) => { 22 };
    (x) => { 23 };
    (y) => { 24 };
    (z) => { 25 };
}

#[macro_export]
macro_rules! ty {
    (@arrow_munch [$($acc:tt)+] -> $($rest:tt)+) => {
        $crate::Type::arrow(
            $crate::ty!(@primary $($acc)+),
            $crate::ty!($($rest)+)
        )
    };
    (@arrow_munch [$($acc:tt)+]) => { $crate::ty!(@primary $($acc)+) };
    (@arrow_munch [$($acc:tt)*] $t:tt $($rest:tt)*) => {
        $crate::ty!(@arrow_munch [$($acc)* $t] $($rest)*)
    };

    // primitive types
    (@primary Integer) => { $crate::Type::integer() };
    (@primary Effect $($arg:tt)+) => { $crate::Type::effect($crate::ty!(@primary $($arg)+)) };

    // interpolated expressions
    (@primary { $e:expr }) => { $e };

    // typevar
    (@primary $v:ident) => { $crate::Type::Var($v) };

    // tuple
    (@primary ( $($inner:tt)* )) => { $crate::ty!(@tuple [] [] $($inner)* @END) };

    // push element on comma
    (@tuple [$($out:expr,)*] [$($acc:tt)+] , $($rest:tt)*) => {
        $crate::ty!(@tuple
            [$($out,)* $crate::ty!($($acc)+), ]
            []
            $($rest)*)
    };

    // on end sentinel, return
    (@tuple [$($out:expr,)*] [] @END) => {
        $crate::Type::tuple(vec![$($out),*])
    };

    // return on leftover element (no trailing comma)
    (@tuple [$($out:expr,)*] [$($acc:tt)+] @END) => {
        $crate::Type::tuple(vec![$($out,)* $crate::ty!($($acc)+)])
    };

    // accumulate current element tokens
    (@tuple [$($out:expr,)*] [$($acc:tt)*] $t:tt $($rest:tt)*) => {
        $crate::ty!(@tuple [$($out,)*] [$($acc)* $t] $($rest)*)
    };

    // entry point
    ($($t:tt)+) => { $crate::ty!(@arrow_munch [] $($t)+) };
}

#[macro_export]
macro_rules! qual {
    // push element on comma
    (@pred_list [$($out:expr,)*] [$($acc:tt)+] , $($rest:tt)*) => {
        $crate::qual!(@pred_list
            [$($out,)* $crate::qual!(@primary $($acc)+), ]
            []
            $($rest)*
        )
    };

    // on end sentinel, return
    (@pred_list [$($out:expr,)*] [] @END) => { vec![$($out),*] };

    // return on leftover element (no trailing comma)
    (@pred_list [$($out:expr,)*] [$($acc:tt)+] @END) => {
        vec![$($out,)* $crate::qual!(@primary $($acc)+)]
    };

    // accumulate current element tokens
    (@pred_list [$($out:expr,)*] [$($acc:tt)*] $t:tt $($rest:tt)*) => {
        $crate::qual!(@pred_list [$($out,)*] [$($acc)* $t] $($rest)*)
    };

    (@primary Num $v:ident) => { $crate::Pred::Num($v) };

    // entry point with multi qualification
    ( ( $($plist:tt)+ ) => $($ty:tt)+) => {{
        $crate::Qual {
            preds: $crate::qual!(@pred_list [] [] $($plist)* @END),
            ty: $crate::ty!($($ty)+)
        }
    }};

    // entry point with solo qualification
    ($class:ident $v:ident => $($ty:tt)+) => {{
        $crate::Qual {
            preds: vec![$crate::qual!(@primary $class $v)],
            ty: $crate::ty!($($ty)+)
        }
    }};

    // entry point with no qualification
    ($($ty:tt)+) => {{
        $crate::Qual {
            preds: vec![],
            ty: $crate::ty!($($ty)+)
        }
    }};
}

#[macro_export]
macro_rules! scheme {
    // entry point with quantification
    (forall $v0:ident $( $v:ident )* . $($qual:tt)+) => {{
        let $v0 = $crate::TypeVar($crate::typevar_index!($v0));
        $( let $v = $crate::TypeVar($crate::typevar_index!($v)); )*
        $crate::Scheme {
            vars: vec![$v0 $(, $v)*],
            qual: $crate::qual!($($qual)+)
        }
    }};

    // entry point with no quantification
    ($($qual:tt)+) => {{
        $crate::Scheme {
            vars: vec![],
            qual: $crate::qual!($($qual)+)
        }
    }};
}
