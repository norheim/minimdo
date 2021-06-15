\ Model cycles
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 assign[z1,4] + 0 assign[z1,5] + 0 assign[z1,6] + 0 assign[x,6]
   + 0 assign[y2,5] + 0 assign[y2,6] + 0 assign[z2,5] + 0 assign[y1,5]
   + 0 assign[y1,6] + C14
Subject To
 equations[4]: assign[4,x] + assign[4,y2] + assign[4,z2] + assign[4,y1]
   = 1
 equations[5]: = 1
 equations[6]: assign[6,f] = 1
 variables[y1]: assign[4,y1] <= 1
 variables[z1]: <= 1
 variables[x]: assign[4,x] <= 1
 variables[y2]: assign[4,y2] <= 1
 variables[z2]: assign[4,z2] <= 1
 variables[f]: assign[6,f] <= 1
Bounds
Binaries
 assign[z1,4] assign[z1,5] assign[z1,6] assign[4,x] assign[4,y2]
 assign[4,z2] assign[4,y1] assign[x,6] assign[y2,5] assign[y2,6]
 assign[z2,5] assign[y1,5] assign[y1,6] assign[6,f]
End
