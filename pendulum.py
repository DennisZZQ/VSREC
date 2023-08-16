# __credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import cv2

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer

DEFAULT_X = 0.5 * np.pi
DEFAULT_Y = 0.0


class PendulumEnv(gym.Env):
    """
       ### Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ### Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ### Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ### Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ### Episode Truncation

    The episode truncates at 200 time steps.

    ### Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```
    gym.make('Pendulum-v1', g=9.81)
    ```

    ### Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, l=1.5, w=0.2):
        self.max_speed = 20
        self.max_torque = 100.0
        self.g = g
        self.m = 1.0
        self.l = l
        self.w = w
        self.a = -(3 * g) / (0.2 * l)
        self.adjustment_1 = 0
        self.adjustment_2 = 0
        self.adjustment_3 = 0
        self.adjustment_4 = 0
        self.c = 0
        self.u = 0

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

        self.screen_dim = 500
        self.screen_h = 640
        self.screen_w = 360
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([4.0, 4.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u, dt=0.005, update_u=False):
        th, thdot = self.state  # th := theta
        g = self.g
        m = self.m
        l = self.l

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        # self.thdotdot = self.a * np.sin(th) + u / (m * l ** 2) + self.adjustment_2 - self.c * thdot
        #
        # self.thdot = thdot + self.thdotdot * dt + self.adjustment_1

        # self.thdot = np.clip(self.thdot, -self.max_speed, self.max_speed)
        # newth = (th + self.thdot * dt)

        if u != 0:
            self.u = u

        newth = th + (thdot + self.adjustment_1) * dt
        self.thdotdot = self.a * np.sin(th) + self.u + self.adjustment_2 - self.c * thdot
        self.thdot = thdot + self.thdotdot * dt

        if update_u:
            self.u = self.u + self.adjustment_3 * dt
            self.c = self.c + self.adjustment_4 * dt

        while newth > np.pi:
            newth -= 2.0 * np.pi

        while newth < -np.pi:
            newth += 2.0 * np.pi

        self.state = np.array([newth, self.thdot])
        self.renderer.render_step()
        return self._get_obs(), -costs, False, False

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        # self.state = self.np_random.uniform(low=low, high=high)
        self.state = [0.25 * np.pi, 0.0]
        self.last_u = None

        self.renderer.reset()
        self.renderer.render_step()
        # if not return_info:
        #     return self._get_obs()
        # else:
        #     return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_h, self.screen_w)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_h, self.screen_w))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_h, self.screen_w))
        self.surf.fill((0, 0, 0))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset_x = self.screen_h // 2
        offset_y = self.screen_w // 2

        rod_length = self.l * scale
        rod_width = self.w * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] - np.pi / 2)  ##

            c = (c[0] + offset_x, c[1] + offset_y)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        # rw, h, w = 15, 150, 150
        # a, b, c = rw / 2, h, w / 2
        # rect1 = [(-a, -c), (-a, c), (a, c), (a, -c)]
        # rect2 = [(-a, c), (a, c), (a + b, -c), (-a + b, -c)]
        # rect3 = [(-a + b, -c), (a + b, -c), (a + b, c), (-a + b, c)]
        #
        # original_coords = [rect1, rect2, rect3]
        # for rec in original_coords:
        #     transformed_coords = []
        #     for c in rec:
        #         c = pygame.math.Vector2(c).rotate_rad(self.state[0] - np.pi / 2)
        #         c = (c[0] + offset_x, c[1] + offset_y)
        #         transformed_coords.append(c)
        #
        #     gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        #     gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        # 首端圆角
        gfxdraw.aacircle(self.surf, offset_x, offset_y, int(0.2 * scale / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset_x, offset_y, int(0.2 * scale / 2), (204, 77, 77)
        )

        # # 末端
        # rod_end = (rod_length, 0)
        # rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] - np.pi / 2)  ##
        # rod_end = (int(rod_end[0] + offset_x), int(rod_end[1] + offset_y))
        #
        # # 末端圆角
        # gfxdraw.aacircle(
        #     self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        # )
        # gfxdraw.filled_circle(
        #     self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        # )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset_x, offset_y, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset_x, offset_y, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
