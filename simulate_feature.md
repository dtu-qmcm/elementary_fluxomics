# Summary

We have a system of differential equations described as follows:

dA/dt = v1 - v2
dQ/dt = v4 - vq_out + vq_in
dP/dt = v5 - vp-out + vp_in
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5
dE/dt = v5 - v2

v1 = constant_parameter
v2 = [A]*[E]*k2f - [EA]*k2r
v3 = [EA]*k3f - [EPQ]*k3r
v4 = [EPQ]*k4f - [Q]*[EP]*k4r
v5 = [EP]*k5f - [E]*[P]*k5r
vq_out = kq_out*[Q]
vq_in = constant_parameter
vp_out = kp_out*[P]
vp_in = constant_parameter

We have an additional constraint that is $\sum ([E], [EA], [EPQ], [EP]) = E_T$
and $E_T$ is also a paramter. The reaction mapping is defined as follows:

A(ab) <-> EPQ(ab)
EPQ(ab) <-> EP(a) + Q(b)

E does not have any labelled atoms.

# Workflow
The script should do the following:
1. Simulate the dynamic system given a set of parameters
2. Using freeflux estimate the timecourse labellings and steady state labelling
given the following:
  - All "A" coming from v1 is labelled 100% on atoms "a" and "b".
  - All "P" and "Q" atoms coming from "vp_in" and "vq_in" are 100% unlabelled
  - The label is a +1 mass difference
  - Reversibilities are given by the forward and reverse reactions that
  are calculated in the provided flux vector
  - Pool sizes are given by the concentrations

