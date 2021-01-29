using DrWatson
@quickactivate "SICMModelingToolkit"
DrWatson.greet()
using ModelingToolkit, Unitful, DifferentialEquations, Optim
using Latexify
using BenchmarkTools, LinearAlgebra, ForwardDiff, Zygote
using Quadrature, StaticArrays


#### section 1.4

@parameters t m
@variables q(t) qv(t)
D = Differential(t)

L(t,x,v) = 1//2 .* m .* dot(v, v)

latexify(L(t,q,qv))

