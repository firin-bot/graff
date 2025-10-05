#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use anyhow::anyhow;
use anyhow::Context as _;
use anyhow::Result;
use derive_more::Deref;
use derive_more::DerefMut;
use derive_more::IntoIterator;
use hashbrown::HashMap;
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
pub struct Scheme {
    pub vars: Vec<TypeVar>,
    pub ty: Type
}

impl Scheme {
    pub fn instantiate(&self, ctx: &mut Context) -> Type {
        let mut subs = SubstitutionMap::with_capacity(self.vars.len());
        for v in &self.vars {
            subs.insert(*v, Type::Var(ctx.fresh_var()));
        }
        self.ty.substitute(&subs)
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
            Self::Instance(instance)   => instance.ty.clone()
        }
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Constant(Value),
    Pure,
    Bind,
    Graph(Box<Graph>),
    Add
}

impl Op {
    pub fn scheme(&self) -> Scheme {
        match self {
            Self::Constant(v) => {
                Scheme {
                    vars: vec![],
                    ty: Type::arrow(
                        Type::unit(),
                        Type::singleton(v.ty())
                    )
                }
            },
            Self::Pure => {
                let a = TypeVar(0);
                Scheme {
                    vars: vec![a],
                    ty: Type::arrow(
                        Type::singleton(Type::Var(a)),
                        Type::singleton(Type::effect(Type::Var(a)))
                    )
                }
            },
            Self::Bind => {
                let a = TypeVar(0);
                let b = TypeVar(1);
                Scheme {
                    vars: vec![a, b],
                    ty: Type::arrow(
                        Type::tuple(vec![
                            Type::effect(Type::Var(a)),
                            Type::arrow(
                                Type::singleton(Type::Var(a)),
                                Type::singleton(Type::effect(Type::Var(b)))
                            )
                        ]),
                        Type::singleton(Type::effect(Type::Var(b)))
                    )
                }
            },
            Self::Graph(g) => g.scheme.clone(),
            Self::Add => {
                let a = TypeVar(0);
                Scheme {
                    vars: vec![a],
                    ty: Type::arrow(
                        Type::tuple(vec![
                            Type::Var(a),
                            Type::Var(a)
                        ]),
                        Type::singleton(Type::Var(a))
                    )
                }
            }
        }
    }

    pub fn instantiate(&self, ctx: &mut Context) -> Instance {
        Instance {
            data: self.clone().into(),
            ty: self.scheme().instantiate(ctx)
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
    pub ty: Type
}

impl Instance {
    pub fn type_of_from_port(&self, port: FromPort) -> Result<&Type> {
        match port {
            FromPort::Instance => Ok(&self.ty),
            FromPort::Index(index) => {
                let (_, o) = self.ty.break_arrow().context("failed to break type as Arrow")?;
                let tuple = o.break_tuple().context("failed to break Arrow output as Tuple")?;
                tuple.get(index).context("output port index out of bounds")
            }
        }
    }

    pub fn type_of_to_port(&self, ToPort(index): ToPort) -> Result<&Type> {
        let (i, _) = self.ty.break_arrow().context("failed to break type as Arrow")?;
        let tuple = i.break_tuple().context("failed to break Arrow output as Tuple")?;
        tuple.get(index).context("input port index out of bounds")
    }

    pub fn evaluate(
        &self,
        graph_inputs: &[Value],
        inputs: BTreeMap<usize, Value>
    ) -> Result<Value> {
        let x = match &self.data {
            InstanceData::Op(op) => match op {
                Op::Constant(val) => Ok(val.clone()),
                Op::Pure          => {
                    let v = inputs.get(&0).context("port 0 unconnected")?.clone();
                    Ok(Value::Effect(Effect {
                        ret_ty: v.ty(),
                        thunk: Rc::new(move || Ok(v.clone()))
                    }))
                },
                Op::Bind          => {
                    let v0 = inputs.get(&0).context("port 0 unconnected")?.clone();
                    let v1 = inputs.get(&1).context("port 1 unconnected")?.clone();
                    match v0 {
                        Value::Effect(effect) => {
                            match v1 {
                                Value::Instance(instance) => {
                                    let graph_inputs = graph_inputs.to_vec();
                                    Ok(Value::Effect(Effect {
                                        ret_ty: instance.type_of_from_port(FromPort::Index(0))?.clone(),
                                        thunk: Rc::new(move || {
                                            let mut inputs: BTreeMap<usize, Value> = Default::default();
                                            inputs.insert(0, effect.run()?);
                                            match instance.evaluate(&graph_inputs, inputs)? {
                                                Value::Effect(effect) => effect.run(),
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
                Op::Graph(g)      => {
                    let subgraph_inputs: Vec<_> = inputs.into_values().collect();
                    g.evaluate(&subgraph_inputs, 0)
                }
                Op::Add           => {
                    let v0 = inputs.get(&0).context("port 0 unconnected")?.clone();
                    let v1 = inputs.get(&1).context("port 1 unconnected")?.clone();
                    if let Value::Integer(i0) = v0 && let Value::Integer(i1) = v1 {
                        Ok(Value::Integer(i0 + i1))
                    } else {
                        Err(anyhow!("invalid type for Add instance"))
                    }
                }
            },
            InstanceData::Input(i) => graph_inputs.get(*i).context("graph input missing").cloned(),
            InstanceData::Output(_) => inputs.get(&0).context("port 0 unconnected").cloned()
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
    outputs: Vec<NodeIndex>
}

impl Graph {
    pub fn new(scheme: Scheme) -> Result<Self> {
        let ctx = Context::with_initial(scheme.vars.len() as u32);
        let mut g: Acyclic<StableDiGraph<Instance, Edge>> = Default::default();

        let (in_ty, out_ty) = scheme.ty.break_arrow().context("failed to break graph scheme as Arrow")?;
        let in_tuple = in_ty.break_tuple().context("failed to break graph input as Tuple")?;
        let out_tuple = out_ty.break_tuple().context("failed to break graph output as Tuple")?;

        let inputs: Vec<_> = in_tuple.iter().enumerate().map(|(i, ty)| {
            g.add_node(Instance {
                data: InstanceData::Input(i),
                ty: Type::arrow(
                    Type::unit(),
                    Type::singleton(ty.clone())
                )
            })
        }).collect();
        let outputs: Vec<_> = out_tuple.iter().enumerate().map(|(i, ty)| {
            g.add_node(Instance {
                data: InstanceData::Output(i),
                ty: Type::arrow(
                    Type::singleton(ty.clone()),
                    Type::unit()
                )
            })
        }).collect();

        Ok(Self {
            ctx,
            g,
            scheme,
            inputs,
            outputs
        })
    }

    pub const fn inner(&self) -> &Acyclic<StableDiGraph<Instance, Edge>> {
        &self.g
    }

    pub fn get_input(&self, index: usize) -> Result<NodeIndex> {
        self.inputs.get(index).copied().with_context(|| format!("input index {} not found", index))
    }

    pub fn get_output(&self, index: usize) -> Result<NodeIndex> {
        self.outputs.get(index).copied().with_context(|| format!("output index {} not found", index))
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
                n.ty = n.ty.substitute(&subs);
            }
        }

        Ok(())
    }

    // XXX: take output port for target node and evaluate only that output
    pub fn evaluate_index(&self, graph_inputs: &[Value], index: NodeIndex, port: FromPort) -> Result<Value> {
        match port {
            FromPort::Instance => Ok(Value::Instance(Box::new(self.g[index].clone()))),
            FromPort::Index(from) => {
                if from > 0 { todo!(); }

                let mut inputs: BTreeMap<usize, Value> = Default::default();
                for edge in self.g.edges_directed(index, Direction::Incoming) {
                    let v = self.evaluate_index(graph_inputs, edge.source(), edge.weight().from)?;
                    inputs.insert(edge.weight().to.0, v);
                }

                self.g[index].evaluate(graph_inputs, inputs)
            }
        }
    }

    pub fn evaluate(&self, graph_inputs: &[Value], index: usize) -> Result<Value> {
        self.evaluate_index(graph_inputs, self.get_output(index).context("output port out of bounds")?, FromPort::Index(0))
    }
}
