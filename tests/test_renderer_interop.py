"""Isolate fatal native-driver errors while alternating live graphics backends."""

import os
import subprocess
import sys
import pytest


def test_pyvista_and_vispy_can_alternate_live_contexts(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("vispy")
    code = r"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions
out=Path(sys.argv[1])
x=torch.linspace(.125,.875,4)
p=ParticleState(positions=torch.cartesian_prod(x,x),supports=torch.ones(16),masses=torch.full((16,),1/16),densities=torch.ones(16),kinds=torch.zeros(16,dtype=torch.int32))
d=DomainDescription(torch.zeros(2),torch.ones(2),torch.zeros(2,dtype=torch.bool),2)
opts=PlottingOptions(markerSize=64,plotDomain=False,showColorBar=False,vMin=0,vMax=1)
pv=visualize(p,d,torch.zeros(16),opts,backend='pyvista')
vp=visualize(p,d,torch.zeros(16),opts,backend='vispy',backendOptions={'jupyter_backend':'image','app_backend':'egl'})
try:
 for i in range(3):
  pv.updateQuantities(torch.full((16,),float(i%2)),redraw=False)
  pv.export(str(out/f'pv_{i}.png'))
  vp.updateQuantities(torch.full((16,),float(i%2)),redraw=False)
  vp.export(str(out/f'vp_{i}.png'))
finally:
 vp.close();pv.close()
# A fresh GLX renderer must work after the EGL surface was destroyed as well.
pv=visualize(p,d,torch.ones(16),opts,backend='pyvista')
try:pv.export(str(out/'reopened.png'))
finally:pv.close()
for name in ('pv','vp'):
 a=np.array(Image.open(out/f'{name}_0.png')).astype(float)
 b=np.array(Image.open(out/f'{name}_1.png')).astype(float)
 assert np.mean(np.abs(a-b))>.1
"""
    env = dict(os.environ, EGL_PLATFORM="surfaceless", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_notebook_frames_and_exports_can_move_between_kernel_threads(tmp_path):
    pytest.importorskip("vispy")
    pytest.importorskip("jupyter_rfb")
    pytest.importorskip("glfw")
    code = r"""
import sys, asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glfw
if not glfw.init():
 sys.exit(77)
glfw.window_hint(glfw.VISIBLE, 0)
probe = glfw.create_window(1, 1, 'probe', None, None)
if not probe:
 sys.exit(77)
glfw.destroy_window(probe)
import numpy as np
import torch
import vispy.app
from PIL import Image
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions
vispy.app.use_app('jupyter_rfb')
out = Path(sys.argv[1])
x = torch.linspace(.125, .875, 4)
p = ParticleState(positions=torch.cartesian_prod(x,x), supports=torch.ones(16), masses=torch.full((16,),1/16), densities=torch.ones(16), kinds=torch.zeros(16,dtype=torch.int32))
d = DomainDescription(torch.zeros(2),torch.ones(2),torch.zeros(2,dtype=torch.bool),2)
s = visualize(p,d,torch.zeros(16),PlottingOptions(markerSize=64,plotDomain=False,showColorBar=False,vMin=0,vMax=1),backend='vispy',backendOptions={'jupyter_backend':'notebook'})
try:
 assert s.fig._backend.css_width == '100%'
 loop = asyncio.get_event_loop()
 callback_errors = []
 loop.set_exception_handler(lambda loop, context: callback_errors.append(context))
 with ThreadPoolExecutor(max_workers=1) as callbacks:
  for i in range(3):
   s.updateQuantities(torch.full((16,),float(i%2)),redraw=True,yieldSeconds=0)
   loop.call_soon(s.fig._backend._rfb_maybe_draw)
   loop.run_until_complete(asyncio.sleep(0))
   assert not callback_errors, callback_errors
   frame = callbacks.submit(s.fig._backend.get_frame).result()
   assert frame is not None and frame.shape[0] > 1
   callbacks.submit(s.export,str(out/f'widget_{i}.png')).result()
 a = np.array(Image.open(out/'widget_0.png')).astype(float)
 b = np.array(Image.open(out/'widget_1.png')).astype(float)
 assert np.mean(np.abs(a-b)) > .1
 s.fig._backend.handle_event({'type':'resize','width':600,'height':400,'ratio':1})
 s.fig._backend._emit_resize_event()
 with ThreadPoolExecutor(max_workers=1) as callbacks:
  callbacks.submit(s.export,str(out/'resized.png')).result()
 assert Image.open(out/'resized.png').size == (600,400)
finally:
 s.close()
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(tmp_path)],
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode == 77:
        pytest.skip("Interactive notebook GLFW context needs an accessible display")
    assert result.returncode == 0, result.stdout + result.stderr


def test_trame_camera_frame_updates_keep_contexts_and_images_valid(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("trame_vtk")
    code = r"""
import io, sys, time, gc, ctypes, ctypes.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
from vtkmodules.vtkWebCore import vtkWebInteractionEvent
from warpSPHCore import ParticleState, DomainDescription
from warpSPHPlotting import visualize, PlottingOptions
from warpSPHPlotting.backends._trame_context import register_trame_window
out = Path(sys.argv[1])
def particles(n):
 x = torch.linspace(.15,.85,n)
 return ParticleState(positions=torch.cartesian_prod(x,x), supports=torch.ones(n*n), masses=torch.full((n*n,),1/(n*n)), densities=torch.ones(n*n), kinds=torch.zeros(n*n,dtype=torch.int32))
p = particles(4)
d = DomainDescription(torch.zeros(2),torch.ones(2),torch.zeros(2,dtype=torch.bool),2)
opts = PlottingOptions(markerSize=36,vMin=0,vMax=1)
s = visualize(p,d,torch.zeros(16),opts,backend='pyvista')
backend = s._backend_instance
# Set up the same shared application as a trame view, without needing a port.
backend._jupyter_backend = 'trame'
guard = backend._trame_context_guard = register_trame_window(s.fig,backend._render_lock)
application, window = guard.application, s.fig.render_window
application.GetObjectIdMap().GetGlobalId(window)
window.GetInteractor().EnableRenderOff()
def nested_capture(_):
 application.InvalidateCache(window)
 application.StillRenderToBuffer(window,0,95)
s.fig.add_on_render_callback(nested_capture)
def current(w):
 # VTK 9.7's EGL IsCurrent flag can remain true after native unbinding.
 if w.IsA('vtkEGLRenderWindow'):
  egl = ctypes.CDLL(ctypes.util.find_library('EGL'))
  egl.eglGetCurrentContext.restype = ctypes.c_void_p
  return bool(egl.eglGetCurrentContext())
 return bool(w.IsCurrent())
def capture(move):
 if move:
  for buttons,x,y in [(1,.4,.5),(1,.43,.53),(0,.43,.53)]:
   event = vtkWebInteractionEvent()
   event.SetButtons(buttons);event.SetX(x);event.SetY(y)
   application.HandleInteractionEvent(window,event)
  event = vtkWebInteractionEvent()
  event.SetScroll(.1)
  application.HandleInteractionEvent(window,event)
 application.InvalidateCache(window)
 image = application.StillRenderToBuffer(window,0,95)
 deadline = time.monotonic()+5
 while application.GetHasImagesBeingProcessed(window):
  assert time.monotonic()<deadline, 'encoder stalled'
  time.sleep(.01)
  image = application.StillRenderToBuffer(window,application.GetLastStillRenderToMTime(),95)
 assert not current(window)
 with Image.open(io.BytesIO(memoryview(image).tobytes())) as decoded:
  return np.array(decoded.convert('RGB'))
try:
 with ThreadPoolExecutor(max_workers=1) as remote:
  for frame in range(6):
   remote.submit(capture,True).result()
   remote.submit(s.fig.render).result()  # toolbar-style direct render
   assert not remote.submit(current,window).result()
   moved = tuple(s.fig.camera_position)
   p = particles(5)
   next_domain = DomainDescription(torch.tensor([-2.,-.5]),torch.tensor([2.,1.5]),d.periodic,2)
   q = torch.linspace(0,1,25) if frame%2 else torch.linspace(1,0,25)
   s.updateQuantities(q,newParticleState=p,newDomain=next_domain,redraw=True,yieldSeconds=0)
   if frame:
    assert tuple(s.fig.camera_position) == moved
   rgb = remote.submit(capture,False).result()
   s.export(str(out/f'trame_{frame}.png'))
   assert not current(window)
   png = np.array(Image.open(out/f'trame_{frame}.png').convert('RGB'))
   assert np.abs(rgb.astype(float)-png).mean()<5
   assert np.count_nonzero(np.ptp(rgb,axis=2)>40)>500
 # Failure must release too, so the next callback can acquire this window.
 try: application.StillRenderToBuffer(window,'invalid-time',95)
 except TypeError: pass
 else: raise AssertionError('expected invalid capture arguments to fail')
 assert not current(window)
 # An unrelated/unregistered window retains the application's normal behavior.
 other = visualize(p,d,torch.ones(25),opts,backend='pyvista')
 try:
  ow = other.fig.render_window
  application.InvalidateCache(ow)
  application.StillRenderToBuffer(ow,0,95)
  assert current(ow)
  ow.ReleaseCurrent()
 finally: other.close()
finally:
 s.close()
assert not guard.windows
assert not hasattr(application,'_warpsph_context_guard')
assert 'StillRenderToBuffer' not in application.__dict__
del s, other, guard, application, window, backend
gc.collect()
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(tmp_path)],
        env=dict(os.environ, MPLBACKEND="Agg", EGL_PLATFORM="surfaceless"),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERR|" not in result.stderr, result.stderr


def test_vispy_streamline_widget_defers_nested_draws_and_serializes_updates(tmp_path):
    pytest.importorskip("vispy")
    pytest.importorskip("jupyter_rfb")
    pytest.importorskip("glfw")
    code = r"""
import asyncio, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import glfw
if not glfw.init(): sys.exit(77)
glfw.window_hint(glfw.VISIBLE,0)
probe=glfw.create_window(1,1,'probe',None,None)
if not probe: sys.exit(77)
glfw.destroy_window(probe)
import numpy as np, torch, warp as wp, vispy.app
from PIL import Image
from warpSPHCore import ParticleState,DomainDescription
from warpSPHPlotting import visualize,PlottingOptions,GridVisualization,Mapping
wp.init();torch.set_num_threads(2);vispy.app.use_app('jupyter_rfb')
out=Path(sys.argv[1])
x=torch.linspace(.1,.9,9);pos=torch.cartesian_prod(x,x)
p=ParticleState(positions=pos,supports=torch.full((81,),.4),masses=torch.full((81,),1/81),densities=torch.ones(81),kinds=torch.zeros(81,dtype=torch.int32))
d=DomainDescription(torch.zeros(2),torch.ones(2),torch.zeros(2,dtype=torch.bool),2)
velocity=torch.stack((-(pos[:,1]-.5),pos[:,0]-.5),dim=1)
s=visualize(p,d,velocity,PlottingOptions(mapping=Mapping.L2),backend='vispy',backendOptions={'jupyter_backend':'notebook'},figsize=(7,5))
backend=s._backend_instance;rfb=s.fig._backend
loop=asyncio.get_event_loop();errors=[]
loop.set_exception_handler(lambda loop,ctx:errors.append(ctx))
try:
 s.updateQuantities(velocity,newOptions={'A':{'gridVisualization':GridVisualization(resolution=32,streamLines=True)}},redraw=True,yieldSeconds=0)
 requested=[]
 def nested(event):
  if requested:return
  requested.append(rfb.get_frame())
  s.show()  # Must neither recurse nor release the outer draw's context.
 s.fig.events.draw.connect(nested,position='first')
 frame=rfb.get_frame()
 s.fig.events.draw.disconnect(nested)
 assert len(requested)==1 and requested[0] is None
 assert frame is not None and frame[:,:,:3].mean()>50
 assert np.count_nonzero(np.ptp(frame[:,:,:3],axis=2)>40)>500
 Image.fromarray(frame).save(out/'streamlines.png')
 # Verify real black streamline pixels over the colored grid.
 line=s.plotStates['A'].stream_visual
 assert line is not None and line.visible
 with backend._canvas_access.use(s.fig):line.visible=False
 without=rfb.get_frame()
 assert np.count_nonzero((frame[:,:,:3].mean(2)<60)&(without[:,:,:3].mean(2)>100))>10
 with backend._canvas_access.use(s.fig):line.visible=True
 entered=threading.Event();release=threading.Event();started=threading.Event();updated=threading.Event()
 paused=[]
 def hold_draw(event):
  if paused:return
  paused.append(True);entered.set()
  assert release.wait(10),'draw coordination timed out'
 def change_frame():
  started.set()
  s.updateTitle('Updated during widget draw')
  s.updateQuantities(velocity*.75,redraw=True,yieldSeconds=0)
  updated.set()
 s.fig.events.draw.connect(hold_draw,position='first')
 try:
  with ThreadPoolExecutor(max_workers=2) as workers:
   drawing=workers.submit(rfb.get_frame)
   assert entered.wait(5)
   update=workers.submit(change_frame)
   assert started.wait(5)
   try: assert not updated.wait(.05),'scene update overlapped an active draw'
   finally: release.set()
   assert drawing.result(timeout=15) is not None
   update.result(timeout=15)
 finally:
  release.set();s.fig.events.draw.disconnect(hold_draw)
 loop.call_soon(rfb._rfb_maybe_draw)
 loop.run_until_complete(asyncio.sleep(0))
 assert not errors,errors
 final=rfb.get_frame()
 assert final is not None and np.count_nonzero(np.ptp(final[:,:,:3],axis=2)>40)>500
 s.export(str(out/'updated.png'))
finally:s.close()
assert rfb.get_frame() is None  # A queued callback after close must be harmless.
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(tmp_path)],
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("Interactive notebook GLFW context needs an accessible display")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EventEmitter loop" not in result.stderr, result.stderr
