% import head mesh
function brain1020mesh(f_headsurf, f_initpoints, p1, p2 ) %input two files, 1 output

if nargin <3
    p1 = 10;
end
if nargin<4
    p2 = 20;
end

%import the vertices (init points) 
initpoints = loadjson(f_initpoints, 'FastArrayParser',0);

%import the headsurf 
headmesh = loadjson(f_headsurf,'FastArrayParser',0);



%compute 10-20
%headsurf=volface(elem(:,1:4));

[landmarks]=brain1020(headmesh.MeshVertex3,headmesh.MeshTri3, initpoints.MeshVertex3, p1,p2,'cztol',1e-8);


%get the mesh 
points = struct2cell(landmarks);
points1 = cell2mat(points);

k = boundary(points1(:,1),points1(:,2),points1(:,3));
h.surf =trimesh(k,points1(:,1),points1(:,2),points1(:,3));
node_brain = h.surf.Vertices;
face_brain = h.surf.Faces;
[n_brain,e_brain] = meshcheckrepair(node_brain,face_brain,'isolated');
blendersavemesh(node_brain,face_brain);
meshdata.MeshVertex3 = n_brain; 
meshdata.MeshTri3 = e_brain;
savejson('',meshdata,'FileName',bpmwpath('brain1020mesh.jmsh'),'ArrayIndent',0);
