import numpy as np


class MultiEnv:
    MAX_STEP_CNT = 200
    CAR_VEL = 4.
    UAV_VEL = 5.
    INIT_UAV_POS = [
        [0., 0.],
        [100., 0.],
        [0., 100.],
        [100., 100.],
        [50., 50.]
    ]
    INIT_CAR_POS = [
        [0., 0.],
        [100., 0.],
        [0., 100.],
        [100., 100.],
        [50., 50.]
    ]

    def __init__(self, uav_cnt, car_cnt, car_policy='random'):
        assert uav_cnt <= 5 and car_cnt <= 5

        self._uav_cnt = uav_cnt
        self._car_cnt = car_cnt

        # [uav_x, uav_y, uav_vx, uav_vy] * uav_cnt + [car_x, car_y, car_vx, car_vy, lost_cnt] * car_cnt
        self.STATE_DIM = self._uav_cnt * 4 + self._car_cnt * 5
        # [vx, vy] * uav_cnt
        self.ACTION_DIM = self._uav_cnt * 2
        self.CAN_SEE_RANGE = 10

        self._uav_pos = np.zeros((uav_cnt, 2))
        self._uav_vel = np.zeros((uav_cnt, 2))

        self._car_pos = np.zeros((car_cnt, 2))
        self._car_vel = np.zeros((car_cnt, 2))

        self._last_car_pos = np.zeros((car_cnt, 3))
        self._last_car_vel = np.zeros((car_cnt, 2))
        self._step_cnt = 0
        self._loss_cnt = 0
        self._get_car_vel = car_policy
        
    def _update(self):
        self._car_pos += self._car_vel
        self._uav_pos += self._uav_vel

    def _visible_info(self):
        visible = {}
        for u in range(self._uav_cnt):
            u_pos = self._uav_pos[u]
            for c in range(self._car_cnt):
                c_pos = self._car_pos[c]
                d = np.dot(u_pos - c_pos, u_pos - c_pos)
                visible[(u, c)] = d <= self.CAN_SEE_RANGE ** 2
        return visible

    def _car_policy(self):
        if self._get_car_vel == 'random':
            self._car_vel = np.random.uniform(-MultiEnv.CAR_VEL, MultiEnv.CAR_VEL, (self._car_cnt, 2))
        elif self._get_car_vel == 'adversary':
            # only support 2 vs 2
            uav_pos = []
            visible = self._visible_info()
            for u in range(self._uav_cnt):
                if visible[(u, 0)] or visible[(u, 1)]:
                    uav_pos.append(self._uav_pos[u])
            uav_pos = np.array(uav_pos)

            if sum(visible.values()) >= 2:
                assert len(uav_pos) > 0
                if len(uav_pos) == 1:
                    uav_pos *= 2
                for c in range(self._car_cnt):
                    c_pos = self._car_pos[c]
                    d0, d1 = np.linalg.norm(uav_pos - c_pos, axis=1)
                    w0, w1 = 1 / (d0 + 0.01), 1 / (d1 + 0.01)
                    weighted_uav_pos = (w0 / (w0 + w1)) * uav_pos[0] + (w1 / (w0 + w1)) * uav_pos[1]
                    direction = c_pos - weighted_uav_pos
                    bigger = np.max(np.abs(direction))
                    self._car_vel[c] = MultiEnv.CAR_VEL * (direction / bigger)

            elif sum(visible.values()) == 1:
                for c in range(self._car_cnt):
                    if visible[(0, c)] or visible[(1, c)]:
                        direction = self._car_pos[c] - uav_pos[0]
                        bigger = np.max(np.abs(direction))
                        self._car_vel[c] = MultiEnv.CAR_VEL * (direction / bigger)
                    else:
                        vel = np.linalg.norm(self._car_vel[c])
                        diff = (self.CAN_SEE_RANGE / vel) * -self._car_vel[c]
                        uav_pos = np.row_stack((uav_pos, diff + self._car_pos[c]))

                        c_pos = self._car_pos[c]
                        d0, d1 = np.linalg.norm(uav_pos - c_pos, axis=1)
                        w0, w1 = 1 / (d0 + 0.01), 1 / (d1 + 0.01)
                        weighted_uav_pos = (w0 / (w0 + w1)) * uav_pos[0] + (w1 / (w0 + w1)) * uav_pos[1]
                        direction = c_pos - weighted_uav_pos
                        bigger = np.max(np.abs(direction))
                        self._car_vel[c] = MultiEnv.CAR_VEL * (direction / bigger)

            else:
                max_vel = [(-MultiEnv.CAR_VEL, -MultiEnv.CAR_VEL),
                           (-MultiEnv.CAR_VEL, MultiEnv.CAR_VEL),
                           (MultiEnv.CAR_VEL, -MultiEnv.CAR_VEL),
                           (MultiEnv.CAR_VEL, MultiEnv.CAR_VEL)]
                max_vel = np.array(max_vel)

                for c in range(self._uav_cnt):
                    seed = np.random.uniform(0, 1)
                    if seed < 0.1:
                        self._car_vel[c] = max_vel[np.random.randint(0, 4)]

            assert self._car_vel.shape == (self._car_cnt, 2)
            self._car_vel = np.clip(self._car_vel, -MultiEnv.CAR_VEL, MultiEnv.CAR_VEL)
        else:
            raise NotImplementedError

    def action_sample(self):
        return np.random.uniform(-MultiEnv.UAV_VEL, MultiEnv.UAV_VEL, (self._uav_cnt, 2)).reshape(-1)

    def _uav_policy(self, v):
        assert v.shape == (self._uav_cnt * 2, )
        for i in range(self._uav_cnt):
            self._uav_vel[i, :] = v[2 * i:2 * i + 2]

    def _get_state_reward(self):
        r = 0
        for c in range(self._car_cnt):
            for u in range(self._uav_cnt):
                uav_pos = self._uav_pos[u]
                car_pos = self._car_pos[c]
                dist_2 = np.dot(uav_pos - car_pos, uav_pos - car_pos)
                if dist_2 < self.CAN_SEE_RANGE ** 2:
                    r += 1
                    self._last_car_pos[c] = np.append(self._car_pos[c], 0.)
                    self._last_car_vel[c] = self._car_vel[c]
                    break
            else:
                self._last_car_pos[c][2] += 1.

        car_pos = self._last_car_pos.reshape((-1,))
        uav_pos = self._uav_pos.reshape((-1,))
        car_vel = self._last_car_vel.reshape((-1,))
        uav_vel = self._uav_vel.reshape((-1,))
        res = np.hstack([uav_pos, uav_vel, car_pos, car_vel])
        # print(res, r)
        return res, r

    def reset(self):
        rho = np.random.uniform(8, 10)
        theta = np.random.uniform(0, 2 * np.pi)
        eps_x = rho * np.cos(theta)
        eps_y = rho * np.sin(theta)
        self._uav_pos = np.array(
            [MultiEnv.INIT_UAV_POS[i] for i in range(self._uav_cnt)]
        )
        self._uav_vel = np.zeros((self._uav_cnt, 2))
        self._car_pos = np.array([
            [
                MultiEnv.INIT_CAR_POS[i][0] + eps_x,
                MultiEnv.INIT_CAR_POS[i][1] + eps_y,
            ] for i in range(self._car_cnt)
        ])
        self._car_vel = np.zeros((self._car_cnt, 2))
        self._car_policy()

        self._last_car_pos = np.append(self._car_pos, [[0.]] * self._car_cnt, axis=1)
        self._last_car_vel = np.copy(self._car_vel)
        self._step_cnt = 0
        self._loss_cnt = 0
        return self._get_state_reward()[0]

    def step(self, action):
        self._step_cnt += 1
        action = np.clip(action, -MultiEnv.UAV_VEL, MultiEnv.UAV_VEL)
        assert len(action) == 2 * self._uav_cnt
        self._update()
        self._uav_policy(action)
        self._car_policy()
        state_, reward = self._get_state_reward()
        if reward == 0:
            self._loss_cnt += 1
        else:
            self._loss_cnt = 0
        return state_, reward, self._step_cnt == MultiEnv.MAX_STEP_CNT or self._loss_cnt > 10, {}
