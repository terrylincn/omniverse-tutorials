import omni
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf, Vt, UsdPhysics
from omni.physx import get_physx_interface
from omni.physx.bindings._physx import SimulationEvent
from omni.physx.scripts.physicsUtils import *
import random

stage = omni.usd.get_context().get_stage()
# set up axis to z
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 0.01)

defaultPrimPath = str(stage.GetDefaultPrim().GetPath())

# light
sphereLight = UsdLux.SphereLight.Define(stage, defaultPrimPath + "/SphereLight")
sphereLight.CreateRadiusAttr(150)
sphereLight.CreateIntensityAttr(30000)
sphereLight.AddTranslateOp().Set(Gf.Vec3f(650.0, 0.0, 1150.0))

# Physics scene
UsdPhysics.Scene.Define(stage, defaultPrimPath + "/physicsScene")


rows = 10
cols = 10
sphereCount = rows*cols
_colors = []

material_scope_path = defaultPrimPath + "/Looks"
UsdGeom.Scope.Define(stage, material_scope_path)

# Trianglemesh materials
for i in range(rows):
    for j in range(cols):
        mtl_path = material_scope_path + "/OmniPBR" + str(i*cols+j)
        mat_prim = stage.DefinePrim(mtl_path, "Material")
        material_prim = UsdShade.Material.Get(stage, mat_prim.GetPath())
        material = UsdPhysics.MaterialAPI.Apply(material_prim.GetPrim())
        mu = 0.0 + ((i * cols + j) % sphereCount) * 0.01
        material.CreateRestitutionAttr().Set(mu)

        if material_prim:
            shader_mtl_path = stage.DefinePrim("{}/Shader".format(mtl_path), "Shader")
            shader_prim = UsdShade.Shader.Get(stage, shader_mtl_path.GetPath())
            if shader_prim:
                shader_out = shader_prim.CreateOutput("out", Sdf.ValueTypeNames.Token)
                material_prim.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
                material_prim.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
                material_prim.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)
                shader_prim.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
                shader_prim.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                shader_prim.SetSourceAssetSubIdentifier("OmniPBR", "mdl")       
                color = Gf.Vec3f(random.random(), random.random(), random.random()) 
                shader_prim.GetPrim().CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(color)  
                _colors.append(color)

# Triangle mesh with multiple materials
path = defaultPrimPath + "/triangleMesh"
_mesh_path = path
mesh = UsdGeom.Mesh.Define(stage, path)

# Fill in VtArrays
points = []
normals = []
indices = []
vertexCounts = []

for i in range(rows):
    for j in range(cols):
        subset = UsdGeom.Subset.Define(stage, path + "/subset" + str(i*cols+j))
        subset.CreateElementTypeAttr().Set("face")
        subset_indices = [i*cols+j]
        rel = subset.GetPrim().CreateRelationship("material:binding", False)
        rel.SetTargets([Sdf.Path(material_scope_path + "/OmniPBR" + str(i*cols+j))])

        points.append(Gf.Vec3f(-stripSize/2 + stripSize * i, -stripSize/2 + stripSize * j, 0.0))
        points.append(Gf.Vec3f(-stripSize/2 + stripSize * (i + 1), -stripSize/2 + stripSize * j, 0.0))
        points.append(Gf.Vec3f(-stripSize/2 + stripSize * (i + 1), -stripSize/2 + stripSize * (j + 1), 0.0))
        points.append(Gf.Vec3f(-stripSize/2 + stripSize * i,-stripSize/2 +  stripSize * (j + 1), 0.0))
        
        for k in range(4):
            normals.append(Gf.Vec3f(0, 0, 1))
            indices.append(k + (i * cols + j) * 4)                

        subset.CreateIndicesAttr().Set(subset_indices)
        vertexCounts.append(4)

mesh.CreateFaceVertexCountsAttr().Set(vertexCounts)
mesh.CreateFaceVertexIndicesAttr().Set(indices)
mesh.CreatePointsAttr().Set(points)
mesh.CreateDoubleSidedAttr().Set(False)
mesh.CreateNormalsAttr().Set(normals)
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
meshCollisionAPI.CreateApproximationAttr().Set("none")

# Sphere material
sphereMaterialpath = defaultPrimPath + "/sphereMaterial"
UsdShade.Material.Define(stage, sphereMaterialpath)
material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(sphereMaterialpath))
material.CreateRestitutionAttr().Set(0.9)

# Spheres
stripSize = 100.0
for i in range(rows):
    for j in range(cols):
        spherePath = "/sphere" + str(i)

        size = 25.0
        position = Gf.Vec3f(i * stripSize, j * stripSize, 250.0)            

        sphere_prim = add_rigid_sphere(stage, spherePath, size, position)

        # Add material
        collisionSpherePath = defaultPrimPath + spherePath
        add_physics_material_to_prim(stage, sphere_prim, Sdf.Path(sphereMaterialpath))

        # apply contact report            
        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(sphere_prim)
        contactReportAPI.CreateThresholdAttr().Set(200000)

collider0 = None
collider1 = None
def _on_simulation_event(event):
    global collider0, collider1, _mesh_path, stage, _colors
    if event.type == int(SimulationEvent.CONTACT_DATA):
        if collider1 == _mesh_path:                
            usdGeom = UsdGeom.Mesh.Get(stage, collider0)
            color = Vt.Vec3fArray([_colors[event.payload['faceIndex1']]])
            usdGeom.GetDisplayColorAttr().Set(color)
    if event.type == int(SimulationEvent.CONTACT_FOUND):
        contactDict = resolveContactEventPaths(event)
        collider0 = contactDict["collider0"]
        collider1 = contactDict["collider1"]
    if event.type == int(SimulationEvent.CONTACT_PERSISTS):
        contactDict = resolveContactEventPaths(event)
        collider0 = contactDict["collider0"]
        collider1 = contactDict["collider1"]

events = get_physx_interface().get_simulation_event_stream()
_simulation_event_sub = events.create_subscription_to_pop(_on_simulation_event)
