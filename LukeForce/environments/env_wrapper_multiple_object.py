# A wrapper that enables supporting multiple objects in one experiment
# from environments.np_physics_env import NpPhysicsEnv


class MultipleObjectWrapper:

    def __init__(self, environment, render, gravity, debug, number_of_cp, gpu_ids, fps, force_multiplier,
                 force_h, state_h, qualitative_size, object_paths):
        # if environment is None:
        #     environment = NpPhysicsEnv
        self.list_of_envs = {}
        self.environment_type, self.number_of_cp, self.force_h, self.state_h, self.qualitative_size = \
            environment, number_of_cp, force_h, state_h, qualitative_size
        self.object_paths = object_paths
        if render:
            assert len(object_paths) == 1, 'if gui only one object can be visualized'
        for obj in self.object_paths:
            self.list_of_envs[obj] = self.environment_type(render=render, object_name=obj,
                                                           object_path=self.object_paths[obj],
                                                           gravity=gravity, debug=debug,
                                                           number_of_cp=number_of_cp, gpu_ids=gpu_ids,
                                                           fps=fps, force_multiplier=force_multiplier,
                                                           force_h=force_h, state_h=state_h,
                                                           qualitative_size=qualitative_size, workers=0)

    def reset(self):
        for obj in self.object_paths:
            self.list_of_envs[obj].reset()

    def get_env_by_obj_name(self, object_name):
        return self.list_of_envs[object_name]

    def init_location_and_apply_force(self, forces, initial_state, object_num=None, list_of_contact_points=None,
                                      no_grad=False):
        assert list_of_contact_points is not None
        if type(initial_state) == dict:
            object_name = initial_state['object_name']
        else:
            object_name = initial_state.object_name
        return self.list_of_envs[object_name].init_location_and_apply_force(forces, initial_state, object_num,
                                                                            list_of_contact_points,
                                                                            no_grad=no_grad)

    def close(self):
        for obj in self.object_paths:
            self.list_of_envs[obj].close()
