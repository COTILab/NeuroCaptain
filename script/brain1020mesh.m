% import head mesh
function brainmesh = brain1020mesh(finput) %input two files, 1 output

%import the headsurf 
headmesh = loadjson(finput,'FastArrayParser',0);
landmarks=brain1020(headmesh.MeshVertex3,headmesh.MeshTri3, headmesh.param.initpoints, headmesh.param.p1,headmesh.param.p2,'cztol',1e-8,'display',0);

%get the mesh 
points = struct2cell(landmarks);
points1 = cell2mat(points);

tet=delaunayn(points1);
face=volface(tet);


meshdata.MeshVertex3 = points1; 
meshdata.MeshTri3 = face; 
brainmesh = savejson('',meshdata,'FileName',ncmwpath('brain1020output.jmsh'),'ArrayIndent',0);
