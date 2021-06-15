\ Model cycles
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 C14
Subject To
 equations[4]: assign[4,z1] + assign[4,y2] + assign[4,x] + assign[4,z2]
   + assign[4,y1] = 1
 equations[5]: assign[5,y2] + assign[5,y1] + assign[5,z1] + assign[5,z2]
   = 1
 equations[6]: assign[6,y2] + assign[6,z1] + assign[6,f] + assign[6,y1]
   + assign[6,x] = 1
 variables[x]: assign[4,x] + assign[6,x] <= 1
 variables[z2]: assign[4,z2] + assign[5,z2] <= 1
 variables[y1]: assign[4,y1] + assign[5,y1] + assign[6,y1] <= 1
 variables[z1]: assign[4,z1] + assign[5,z1] + assign[6,z1] <= 1
 variables[y2]: assign[4,y2] + assign[5,y2] + assign[6,y2] <= 1
 variables[f]: assign[6,f] <= 1
Bounds
Binaries
 assign[4,z1] assign[4,y2] assign[4,x] assign[4,z2] assign[4,y1]
 assign[5,y2] assign[5,y1] assign[5,z1] assign[5,z2] assign[6,y2]
 assign[6,z1] assign[6,f] assign[6,y1] assign[6,x]
End
