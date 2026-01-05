# Dynamical Systems Computational Exploration

During my third semester I took a class on Topological Dynamics. This is a small exploration done on Python to study some maps to try and interpret their behaviour. 

The repository contains a file for live orbits of the map and allows the visualization of the effect of bifurcations 'ActiveDyn.py'. 


The following are small descriptions of each file and their functionality:

'graph_attract.py':

## Interactive Attractor Plotter
   * asks for N (number of points)
   * lets the user choose a map from a menu
   * asks for map parameters (showing defaults)
   * computes the orbit and plots the SRB density
 
  Dependencies: numpy, matplotlib


'bifurcation.py':

## Attractor Plotting + Bifurcation Diagrams

  * All maps (Henon, Logistic, Ikeda, Duffing, …)
  * Interactive: choose N, map, parameters, then
    – plot attractor (SRB density)  OR
    – plot bifurcation diagram for one parameter

  Dependencies: numpy, matplotlib
  Install with: pip install numpy matplotlib

'active_dyn.py':

## Dynamical‑Systems Lab – full version with Duffing map
Includes:
   Henon, Ikeda, Standard, Logistic,
   Lozi, DeJong (damped), Damped (variant of standard map),
   Skew‑tent, and the Duffing map 


'papo.py'

## Holomorphic‑Dynamics Explorer – high‑quality,
    pre‑computed picture + pygame UI, CPU‑only but
    accelerated with Numba (32‑bit floats).
