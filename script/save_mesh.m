% import head mesh
function brainmesh = save_mesh(finput,fname) %

%import the headsurf 
mesh = loadjson(finput,'FastArrayParser',0);

savejd([],mesh, fname);
%get the mesh 
