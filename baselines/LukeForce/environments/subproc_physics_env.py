from environments.base_env import BaseBulletEnv
from environments.physics_env import PhysicsEnv

import pybullet_data
from utils.transformations import quaternion_normal2bullet, quaternion_bullet2normal

import os
import time
import gym
import numpy as np
import torch
import multiprocessing as mp
from utils.obj_util import obtain_all_vertices_from_obj
from utils.quaternion_util import quaternion_to_rotation_matrix
from utils.environment_util import EnvState
from utils.constants import OBJECT_TO_SCALE, CONTACT_POINT_MASK_VALUE, GRAVITY_VALUE
from utils.multi_process import CloudpickleWrapper, clear_mpi_env_vars

from environments.base_env import BaseBulletEnv


def give_env_fns(num, render, object_name, object_path, gravity, debug, number_of_cp, gpu_ids, fps,
                 force_multiplier, force_h, state_h, qualitative_size, workers=0):
    def make_env(seed):
        def _thunk():
            env = PhysicsEnv(render=render, object_name=object_name, object_path=object_path, gravity=gravity,
                             debug=debug, number_of_cp=number_of_cp, gpu_ids=gpu_ids, fps=fps,
                             force_multiplier=force_multiplier, force_h=force_h, state_h=state_h,
                             qualitative_size=qualitative_size, workers=workers)
            env.seed(seed)
            return env
        return _thunk
    fns = [make_env(i) for i in range(num)]
    return fns


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, data):
        forces, initial_state, object_num, list_of_contact_points = \
            data['forces'], data['initial_state'], data['object_num'], data['list_of_contact_points']
        # use this to call init_location_and_apply_force
        current_state, list_of_force_success, list_of_force_location = \
            env.init_location_and_apply_force(forces=forces, initial_state=initial_state,
                                              object_num=object_num,
                                              list_of_contact_points=list_of_contact_points)
        # transfer to picklable objects.
        current_state = current_state.to_dict()
        list_of_force_location = [ele.tolist() for ele in list_of_force_location]

        return current_state, list_of_force_success, list_of_force_location

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, data) for env, data in zip(envs, data)])
            elif cmd == 'reset':
                for env in envs:
                    env.reset()
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


# run multiple envs of one object parallelly.
class SubprocPhysicsEnv:
    def __init__(self, args, object_path, object_name, context='spawn', nproc=35):
        self.obj_path, self.obj_name = object_path, object_name

        # multienv settings
        self.closed, self.waiting = False, False
        self.nremotes = nproc
        env_fns = give_env_fns(num=self.nremotes, render=args.render, object_name=object_name,
                               object_path=object_path,
                               gravity=args.gravity, debug=args.debug,
                               number_of_cp=args.number_of_cp, gpu_ids=args.gpu_ids,
                               fps=args.fps, force_multiplier=args.force_multiplier,
                               force_h=args.force_h, state_h=args.state_h,
                               qualitative_size=args.qualitative_size, workers=0)
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker,
                               args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            # with clear_mpi_env_vars():
            p.start()
        for remote in self.work_remotes:
            remote.close()

    # batch functions.
    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def batch_async(self, batch_data):
        self._assert_not_closed()
        batch_data = np.array_split(batch_data, self.nremotes)
        for remote, data in zip(self.remotes, batch_data):
            remote.send(('step', data))
        self.waiting = True

    def batch_wait(self):
        self._assert_not_closed()
        batch_results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return batch_results

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))

    def batch_init_locations_and_apply_force(self, batch_data):
        self.batch_async(batch_data=batch_data)
        return self.batch_wait()

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send('close', None)
        for p in self.ps:
            p.join()

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True
