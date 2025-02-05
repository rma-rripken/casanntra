
# Training data for ANNs

These are the conventions as per CalSim ANN training.

| Variable Name    | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| datetime         | year-month-day  (all values are daily-averaged)|
| model            | Modeling platform used for this dataset (e.g. SCHISM, DSM2, RMA)   |
| scene            | Scenario being modeled - geographically (e.g. Baseline, Suisun Suite, Cache Suite)   |
| case             | Case number being referenced which affects time-series inputs   |
| northern_flow    | Sacramento (below Feather) + American + Yolo Bypass and Toe Drain + Mokelumne + Cosumnes + Calaveras - North Bay Aqueduct (cfs) |
| sac_flow         | Sacramento River Inflow (below Feather) (cfs) |
| sjr_flow         | San Joaquin River Inflow (cfs) |
| exports          | Banks + Jones + CCC Rock Slough + CCC Middle/Old + CCC Victoria (cfs)|
| delta_cu         | Net Delta Consumptive Use (cfs) |
| ndo              | Net Delta Outflow as calculated by boundary inflows (cfs) |
| dcc              | Delta Cross Channel gate operations (0=Closed, 1=Open) |
| smscg            | Suisun Marsh Salinity Control Gate operations (0=Open, 1=Tidally Operated) |
| sf_tidal_energy  | Tidal energy of SFFPX station/boundary. Calculated over stage in feet. < (z- <z>)^2 >, where <> is a low-pass filter |
| sf_tidal_filter  | Tidal energy of SFFPX station/boundary. Calculated over stage in feet. <z>, where <> is a low-pass filter |
| x2               | X2 calculated from model EC results |
| ec locations     | trp,wci,vcu,rsl,old,rri2,bdt,lps,snc,dsj,bdl,nsl2,vol,tss,sss,oh4,vcu,god,bac,hol,mtz,tms,gzl,rsl |
