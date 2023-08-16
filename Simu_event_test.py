import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from pendulum import PendulumEnv
from sklearn.neighbors import KDTree
import threading
import asyncio

class EKF:
    def __init__(self, initial_state, l, g):
        self.H = np.eye(2)
        self.Q = 1e-3 * np.eye(2)
        self.R = 1e-2 * np.eye(2)
        self.P = np.eye(2)
        self.X_est = np.array(initial_state)
        self.a = -l / g
        self.P_pri = None
        self.A = None
        self.X_pri = None

    def compute_A(self, delta_t):
        return np.array([[1, delta_t], [self.a * np.cos(self.X_est[0]) * delta_t, 1]])

    def compute_X_prior(self, delta_t):
        newth = self.X_est[0] + self.X_est[1] * delta_t
        thdot = self.X_est[1] + self.a * np.sin(self.X_est[0]) * delta_t
        return np.array([newth, thdot])

    def compute_P_pri(self):
        return np.dot(np.dot(self.A, self.P), self.A.T + self.Q)

    def update(self, delta_t, Z_k):
        self.A = self.compute_A(delta_t)
        self.P_pri = self.compute_P_pri()
        self.X_pri = self.compute_X_prior(delta_t)
        S = self.H @ self.P_pri @ self.H.T
        K_k = self.P_pri @ self.H.T @ np.linalg.inv(S)
        self.X_est = self.X_pri + K_k @ (Z_k - self.H @ self.X_pri)

        return self.X_est[0]


class EventCamera:
    def __init__(self):
        self.masks = None
        self.EKF = None
        self.real = PendulumEnv(render_mode='rgb_array',l=1.5, g=2)
        self.sim = PendulumEnv(render_mode='rgb_array',l=1.5)
        self.event_sim = None
        self.real.c = 0.4
        self.event_real = None
        self.E_real = None
        self.E_sim = None
        self.interval = 0.001
        self.time = time
        # self.K1 = - 150
        # self.K2 = - 500
        # self.K3 = - 6000

        # radius
        self.K1 = - 50
        self.K2 = - 400
        self.K3 = - 1000
        self.K4 = - 0.1
        self.K5 = - 0.1
        self.window = np.ones([9, 9])

        self.offset = np.array([self.real.screen_h // 2, -self.real.screen_w // 2]) #h:640;w:360

    def update_events(self, S, gray_img, c=0.5, filter=False):
        c1 = 2 ** c
        c2 = 2 ** (-c)
        E_pos = ((gray_img + 25) / (S + 25) > c1) + 0
        E_neg = ((gray_img + 25) / (S + 25) < c2) + 0

        if filter:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            opened1 = cv2.morphologyEx(E_neg.astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=1)
            opened2 = cv2.morphologyEx(E_pos.astype('uint8'), cv2.MORPH_OPEN, kernel, iterations=1)
            E = opened2.astype(int) - opened1.astype(int)
            img2 = 255 * np.ones((np.shape(E)[0], np.shape(E)[1], 3))

            img2[opened2 > 0] = (0, 0, 255)
            img2[opened1 > 0] = (255, 0, 0)

            new_S = (1 - opened2 - opened1) * S + (opened1 + opened2) * gray_img
            return new_S, E, img2

        E = E_pos - E_neg
        new_S = (1 - E_pos - E_neg) * S + (E_neg + E_pos) * gray_img

        img = 255 * np.ones((np.shape(E)[0], np.shape(E)[1], 3))

        img[E_pos > 0] = (0, 0, 255)
        img[E_neg > 0] = (255, 0, 0)
        return new_S, E, img

    def creat_rectangle_mask(self):
        mask_sizes = [(135, 240), (90, 160), (45, 80)]
        masks = []

        # Generate the rectangular masks
        for mask_size in mask_sizes:
            mask_image = np.ones((360, 640), dtype=np.uint8)
            start_x = 270 - mask_size[0] // 2
            end_x = start_x + mask_size[0]
            start_y = 160 - mask_size[1] // 2
            end_y = start_y + mask_size[1]
            mask_image[start_x:end_x, start_y:end_y] = 0
            masks.append(mask_image)
        return masks

    def creat_ring_masks(self, bitmap, ring_width):
        height, width = bitmap.shape
        center_x = width // 2
        center_y = height // 2
        max_radius = min(height, width) // 2

        outer_radii = np.arange(40, max_radius + 1, ring_width)
        inner_radii = outer_radii - ring_width

        y_indices, x_indices = np.indices((height, width))

        distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

        masks = []

        for inner_radius, outer_radius in zip(inner_radii, outer_radii):
            mask = np.logical_and(inner_radius <= distances, distances < outer_radius).astype(np.uint8)
            masks.append((mask, outer_radius - ring_width//2))

        return masks

    def nearest_neighbor_mask(self, E_real, E_sim):
        vecx = 0
        vecy = 0
        simx = 0
        simy = 0

        if not np.isnan(E_real.any()) and not np.isnan(E_sim.any()):
            center_x = self.sim.screen_w / 2
            center_y = self.sim.screen_h / 2
            num_of_points = len(self.masks)

            for mask, r in self.masks:
                # t = time.time()
                # real_masked = np.logical_and(E_real, mask)
                # sim_masked = np.logical_and(E_sim, mask)
                real_masked = E_real * mask
                sim_masked = E_sim * mask
                # print(time.time() - t, 1)
                real_center_x, real_center_y = np.mean(np.where(real_masked != 0), axis=1)
                sim_center_x, sim_center_y = np.mean(np.where(sim_masked != 0), axis=1)
                if np.isnan(real_center_x) or np.isnan(sim_center_x):
                    continue
                cv2.circle(self.event_sim, (int(sim_center_y), int(sim_center_x)), 5,
                           (128, 0, 128), -1)
                cv2.circle(self.event_real, (int(real_center_y), int(real_center_x)), 5,
                           (0, 64, 128), -1)
                simx = simx + sim_center_x / num_of_points
                simy = simy + sim_center_y / num_of_points
                vecx = vecx + (sim_center_x - real_center_x) / (num_of_points * r)
                vecy = vecy + (sim_center_y - real_center_y) / (num_of_points * r)

            simx = (simx - center_x) / self.sim.screen_w
            simy = (simy - center_y) / self.sim.screen_h

        diff_corrected = (-simy * vecx + simx * vecy)
        return diff_corrected


    def draw_center(self):
        events_coord = np.where(self.E_real != 0)
        if len(events_coord[0]) >= 100:
            center_x, center_y = np.mean(np.where(self.E_real != 0), axis=1)
            cv2.circle(self.event_real, (int(center_y), int(center_x)), 20, (0, 0, 255), -1)

    def compute_th(self, E):
        center_real = np.mean(np.where(E != 0), axis=1) + [1, 1]
        center_real[0], center_real[1] = center_real[1], - center_real[0]
        center_real = center_real - self.offset
        # center_sim = np.mean(np.where(self.E_sim != 0), axis=1) - self.offset
        TheNorm = np.linalg.norm(center_real)
        th = np.arccos(np.dot([0, -1], center_real) / TheNorm)
        rho = np.arcsin(np.cross([0, -1], center_real) / TheNorm)
        th = th if rho >= 0 else -th
        return th

    def nearest_neighbor(self, events, sim_events):
        events = events / [self.sim.screen_w, self.sim.screen_h]
        sim_events = sim_events / [self.sim.screen_w, self.sim.screen_h]
        if 100 < len(events) < 10000 and len(sim_events) > 100:
            tree = KDTree(events)
            vecx = 0
            vecy = 0
            simx = 0
            simy = 0
            for i in range(50):
                pick = np.random.choice(len(sim_events))
                a = sim_events[pick][0]
                b = sim_events[pick][1]
                point = np.array([[a, b]])
                _, index = tree.query(point, k=1, return_distance=True)
                index = index[0][0]
                # cv2.circle(self.event_real, (int(tree.data[index][1]), int(tree.data[index][0])), 5, (0, 0, 255), -1)
                # cv2.circle(self.event_sim, (b, a), 5, (0, 0, 255), -1)

                simx = simx + point[0][0] / 50
                simy = simy + point[0][1] / 50
                vecx = vecx + (point[0][0] - tree.data[index][0]) / 50
                vecy = vecy + (point[0][1] - tree.data[index][1]) / 50

            simx -= 0.5
            simy -= 0.5

            # diff_corrected = - simy * vecx + simx * vecy
            diff_corrected = np.sign(vecy) * ((vecy ** 2 + vecx ** 2) ** 0.5)
        else:
            diff_corrected = 0

        return diff_corrected

    def nearest_neighbor_in_radius_polar(self, E_real, E_sim):
        num_of_points = 30
        center_x = self.sim.screen_w / 2
        center_y = self.sim.screen_h / 2
        pos_events = np.transpose(np.nonzero(E_real == 1)) + [1, 1]
        neg_events = np.transpose(np.nonzero(E_real == -1)) + [1, 1]
        pos_sim_events = np.transpose(np.nonzero(E_sim == 1)) + [1, 1]
        neg_sim_events = np.transpose(np.nonzero(E_sim == -1)) + [1, 1]

        if 50 < len(pos_events) < 5000 and len(pos_sim_events) > 100:
            pos_x_coords = pos_events[:, 0] - center_x
            pos_y_coords = pos_events[:, 1] - center_y
            pos_rho, pos_theta = np.sqrt(pos_x_coords ** 2 + pos_y_coords ** 2), np.arctan2(pos_y_coords, pos_x_coords)
            pos_polar_coords = np.column_stack((pos_rho, pos_theta))
            pos_kdtree = KDTree(pos_polar_coords)
            vecx = 0
            vecy = 0
            simx = 0
            simy = 0
            pos_count = 0
            for i in range(num_of_points//2):
                pick = np.random.choice(len(pos_sim_events))
                a = pos_sim_events[pick][0] - center_x
                b = pos_sim_events[pick][1] - center_y
                point = np.array([[np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)]])
                _, index = pos_kdtree.query(point, k=1)
                index = index[0][0]
                # print(point[0])
                if abs(pos_kdtree.data[index][0] - point[0][0]) >= 5:
                    continue
                pos_count += 1
                c = int(pos_kdtree.data[index][0] * np.cos(pos_kdtree.data[index][1]))
                d = int(pos_kdtree.data[index][0] * np.sin(pos_kdtree.data[index][1]))

                cv2.circle(self.event_sim, (int(b + center_y), int(a + center_x)), 5,
                           (128, 0, 128), -1)
                cv2.circle(self.event_real, (int(d + center_y), int(c + center_x)), 5,
                           (128, 0, 128), -1)

                simx = simx + a
                simy = simy + b
                vecx = vecx + (a - c) / pos_kdtree.data[index][0]
                vecy = vecy + (b - d) / pos_kdtree.data[index][0]

            neg_count = 0
            if 50 < len(neg_events) < 5000 and len(neg_sim_events) > 100:
                neg_x_coords = neg_events[:, 0] - center_x
                neg_y_coords = neg_events[:, 1] - center_y
                neg_rho, neg_theta = np.sqrt(neg_x_coords ** 2 + neg_y_coords ** 2), np.arctan2(neg_y_coords,
                                                                                                neg_x_coords)
                neg_polar_coords = np.column_stack((neg_rho, neg_theta))
                neg_kdtree = KDTree(neg_polar_coords)
                for j in range(num_of_points//2):
                    pick = np.random.choice(len(neg_sim_events))
                    a = neg_sim_events[pick][0] - center_x
                    b = neg_sim_events[pick][1] - center_y
                    point = np.array([[np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)]])
                    _, index = neg_kdtree.query(point, k=1)
                    index = index[0][0]
                    if abs(neg_kdtree.data[index][0] - point[0][0]) >= 5:
                        continue
                    neg_count += 1
                    c = int(neg_kdtree.data[index][0] * np.cos(neg_kdtree.data[index][1]))
                    d = int(neg_kdtree.data[index][0] * np.sin(neg_kdtree.data[index][1]))

                    cv2.circle(self.event_sim, (int(b + center_y), int(a + center_x)), 5,
                               (0, 64, 128), -1)
                    cv2.circle(self.event_real, (int(d + center_y), int(c + center_x)), 5,
                               (0, 64, 128), -1)

                    simx = simx + a
                    simy = simy + b
                    vecx = vecx + (a - c) / neg_kdtree.data[index][0]
                    vecy = vecy + (b - d) / neg_kdtree.data[index][0]

            count = neg_count + pos_count
            if count == 0:
                return 0
            simx /= (self.sim.screen_w * count)
            simy /= (self.sim.screen_h * count)
            vecx /= count
            vecy /= count

            diff_corrected = -simy * vecx + simx * vecy
            return diff_corrected
        else:
            return 0

    def nearest_neighbor_in_radius(self, E_real, E_sim):
        num_of_points = 30
        pos_events = np.transpose(np.nonzero(E_real == 1)) + [1, 1]
        neg_events = np.transpose(np.nonzero(E_real == -1)) + [1, 1]
        pos_sim_events = np.transpose(np.nonzero(E_sim == 1)) + [1, 1]
        neg_sim_events = np.transpose(np.nonzero(E_sim == -1)) + [1, 1]
        center_x = self.sim.screen_w / 2
        center_y = self.sim.screen_h / 2
        # t = time.perf_counter()
        if 50 < len(pos_events) < 5000 and len(pos_sim_events) > 100:
            vecx = 0
            vecy = 0
            simx = 0
            simy = 0
            pos_distances = np.sqrt((pos_events[:, 0] - center_x) ** 2 + (pos_events[:, 1] - center_y) ** 2)
            neg_distances = np.sqrt((neg_events[:, 0] - center_x) ** 2 + (neg_events[:, 1] - center_y) ** 2)
            for i in range(num_of_points//2):
                pick = np.random.choice(len(pos_sim_events))
                x1 = pos_sim_events[pick][0]
                y1 = pos_sim_events[pick][1]
                d1 = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)
                radius = round(d1)
                filtered_coords = pos_events[np.round(pos_distances) == radius]
                if len(filtered_coords) == 0:
                    continue
                filtered_distances = np.sqrt(
                    (filtered_coords[:, 0] - x1) ** 2 + (filtered_coords[:, 1] - y1) ** 2)

                nearest_index = np.argmin(filtered_distances)

                nearest_pixel_x = filtered_coords[nearest_index, 0]
                nearest_pixel_y = filtered_coords[nearest_index, 1]
                # print(nearest_pixel_x)
                cv2.circle(self.event_sim, (y1, x1), 5,
                           (128, 0, 128), -1)
                cv2.circle(self.event_real, (nearest_pixel_y, nearest_pixel_x), 5,
                           (128, 0, 128), -1)

                simx = simx + x1 / num_of_points
                simy = simy + y1 / num_of_points
                vecx = vecx + (x1 - nearest_pixel_x) / (num_of_points * radius)
                vecy = vecy + (y1 - nearest_pixel_y) / (num_of_points * radius)

            if 50 < len(neg_events) < 5000 and len(neg_sim_events) > 100:
                for i in range(num_of_points//2):
                    pick = np.random.choice(len(neg_sim_events))
                    x1 = neg_sim_events[pick][0]
                    y1 = neg_sim_events[pick][1]
                    d1 = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)
                    radius = round(d1)
                    filtered_coords = neg_events[np.round(neg_distances) == radius]
                    if len(filtered_coords) == 0:
                        continue
                    filtered_distances = np.sqrt(
                        (filtered_coords[:, 0] - x1) ** 2 + (filtered_coords[:, 1] - y1) ** 2)

                    nearest_index = np.argmin(filtered_distances)

                    nearest_pixel_x = filtered_coords[nearest_index, 0]
                    nearest_pixel_y = filtered_coords[nearest_index, 1]
                    cv2.circle(self.event_sim, (y1, x1), 5,
                               (0, 64, 128), -1)
                    cv2.circle(self.event_real, (nearest_pixel_y, nearest_pixel_x), 5,
                               (0, 64, 128), -1)

                    simx = simx + x1 / num_of_points
                    simy = simy + y1 / num_of_points
                    vecx = vecx + (x1 - nearest_pixel_x) / (num_of_points * radius)
                    vecy = vecy + (y1 - nearest_pixel_y) / (num_of_points * radius)

            simx = (simx - center_x) / self.sim.screen_w
            simy = (simy - center_y) / self.sim.screen_h
            vecy = vecy
            vecx = vecx

            diff_corrected = (- simy * vecx + simx * vecy)
            # print(time.perf_counter() - t, 1)
            # if diff_corrected > 0:
                # cv2.circle(self.event_sim, (100, 100), 5,
                #            (0, 64, 128), -1)
            # else:
                # cv2.circle(self.event_sim, (100, 100), 5,
                #            (128, 0, 128), -1)
            return diff_corrected
        else:
            cv2.circle(self.event_sim, (100, 100), 5,
                       (0, 0, 255), -1)
            return 0

    def density_neignbor(self, E_real, E_sim):

        if self.E_real.any() != 0 and self.E_sim.any() != 0:
            non_zero_counts_real = cv2.filter2D(E_real, ddepth=-1, kernel=self.window)

            a = np.max(non_zero_counts_real)
            first_index = np.mean(np.where(non_zero_counts_real == a), axis=1)/ [self.sim.screen_w, self.sim.screen_h]
            second_index = np.mean(np.where(non_zero_counts_real == a//2), axis=1)/ [self.sim.screen_w, self.sim.screen_h]

            b = np.min(non_zero_counts_real)
            third_index = np.mean(np.where(non_zero_counts_real == b), axis=1) / [self.sim.screen_w, self.sim.screen_h]
            forth_index = np.mean(np.where(non_zero_counts_real == b // 2), axis=1) / [self.sim.screen_w,
                                                                                        self.sim.screen_h]

            non_zero_counts_sim = cv2.filter2D(E_sim, ddepth=-1, kernel=self.window)

            a_sim = np.max(non_zero_counts_sim)
            first_index_sim = np.mean(np.where(non_zero_counts_sim == a_sim), axis=1)/ [self.sim.screen_w, self.sim.screen_h]
            second_index_sim = np.mean(np.where(non_zero_counts_sim == a_sim // 2), axis=1)/ [self.sim.screen_w, self.sim.screen_h]

            b_sim = np.min(non_zero_counts_sim)
            third_index_sim = np.mean(np.where(non_zero_counts_sim == b_sim), axis=1) / [self.sim.screen_w,
                                                                                         self.sim.screen_h]
            forth_index_sim = np.mean(np.where(non_zero_counts_sim == b_sim // 2), axis=1) / [self.sim.screen_w,
                                                                                               self.sim.screen_h]

            simx = first_index_sim[0]/3 + second_index_sim[0]/6 + third_index_sim[0]/3 + forth_index_sim[0]/6
            simy = first_index_sim[1] / 3 + second_index_sim[1] / 6 + third_index_sim[1]/3 + forth_index_sim[1]/6
            realx = first_index[0] / 3 + second_index[0] / 6 + third_index[0]/3 + forth_index[0]/6
            realy = first_index[1] / 3 + second_index[1] / 6 + third_index[1]/3 + forth_index[1]/6
            if not np.isnan(simy) and not np.isnan(simx):
                cv2.circle(self.event_sim, (int(self.sim.screen_h * simy), int(self.sim.screen_w * simx)), 5, (255, 0, 0), -1)
                cv2.circle(self.event_real, (int(self.sim.screen_h * realy), int(self.sim.screen_w * realx)), 5, (255, 0, 0), -1)
                vecx = simx - realx
                vecy = simy - realy
                simx -= 0.5
                simy -= 0.5

                diff_corrected = -simy * vecx + simx * vecy
            else:
                diff_corrected = 0
        else:
            diff_corrected = 0
        return diff_corrected

    def update_length(self, dt):

        center_real = np.mean(np.where(self.E_real != 0), axis=1) + [1, 1]
        center_real[0], center_real[1] = center_real[1], - center_real[0]
        center_real = center_real - self.offset

        center_sim = np.mean(np.where(self.E_sim != 0), axis=1) + [1, 1]
        center_sim[0], center_sim[1] = center_sim[1], - center_sim[0]
        center_sim = center_sim - self.offset

        length_diff = np.linalg.norm(center_sim) - np.linalg.norm(center_real)
        if not np.isnan(length_diff):
            self.sim.l += self.K4 * length_diff * dt

    def update_width(self, dt):
        pos_center_real = np.mean(np.where(self.E_real == 1), axis=1) + [1, 1]
        neg_center_real = np.mean(np.where(self.E_real == -1), axis=1) + [1, 1]
        width_real = np.linalg.norm(pos_center_real - neg_center_real)

        pos_center_sim = np.mean(np.where(self.E_sim == 1), axis=1) + [1, 1]
        neg_center_sim = np.mean(np.where(self.E_sim == -1), axis=1) + [1, 1]
        width_sim = np.linalg.norm(pos_center_sim - neg_center_sim)

        width_diff = width_sim - width_real
        if not np.isnan(width_diff):
            self.sim.w += self.K5 * width_diff * dt

    def generate_torque(self):
        u = float(self.desired_pos) - self.sim.state[0]
        if u > np.pi:
            u = u - (2 * np.pi)
        elif u < -np.pi:
            u = u + (2 * np.pi)
        else:
            u = u
        u = u + np.pi


    async def read_user_input(self):
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input)
            if user_input != 'q':
                # Invoke the callback function with the user input as an argument
                self.desired_pos = user_input
            else:
                self.flag = 1

    def event_camera_sim(self):
        self.real.reset()
        self.real.state = [-0.25 * np.pi, 0.0]
        pre_th = self.real.state[0]
        self.occlusion = self.creat_rectangle_mask()

        self.EKF = EKF([-0.4 * np.pi, 0.0], g=10.0, l=1.0)
        S_real = cv2.cvtColor(np.squeeze(self.real.render()), cv2.COLOR_BGR2GRAY).astype(int)
        # S_real = S_real * self.occlusion[0]
        self.masks = self.creat_ring_masks(S_real, 20)


        self.sim.reset()
        self.sim.w = 0.6
        S_sim = cv2.cvtColor(np.squeeze(self.sim.render()), cv2.COLOR_BGR2GRAY).astype(int)
        time1 = time.time()
        time2 = time.time()

        real_ths = []
        sim_ths = []
        est_ths = []
        start = time.time()
        time_step = []
        count = 0
        length = []
        d_pos = []
        #
        # self.input_thread = threading.Thread(target=asyncio.run, args=(self.read_user_input(),))
        # self.input_thread.start()

        while True:
            if self.time.time() - time1 >= 0.02:
                dt = time.time() - time1
                time1 = time.time()
                observation, reward, done, info = self.real.step([0], dt)
                # self.real.state[0] = self.real.state[0] + 1.5 * dt
                # if self.real.state[0] > np.pi:
                #     self.real.state[0] = self.real.state[0] - (2 * np.pi)
                # elif self.real.state[0] < -np.pi:
                #     self.real.state[0] = self.real.state[0] + (2 * np.pi)
                # else:
                #     self.real.state[0] = self.real.state[0]
                #
                # self.real.state[1] = 1.5

                # self.real.renderer.render_step()
                observation, reward, done, info = self.sim.step([0], dt, True)
                # est_th = self.EKF.update(dt, self.sim.state)
                real_ths.append(self.real.state[0])
                sim_ths.append(self.sim.state[0])
                time_step.append(time.time() - start)
                count += 1

                gray_img_real = cv2.cvtColor(np.squeeze(self.real.render()), cv2.COLOR_BGR2GRAY).astype(int)
                # gray_img_real = gray_img_real * self.occlusion[0]

                S_real, self.E_real, self.event_real = self.update_events(S_real, gray_img_real)


                # for synthetic pendulum
                gray_img_sim = cv2.cvtColor(np.squeeze(self.sim.render()), cv2.COLOR_BGR2GRAY).astype(int)
                S_sim, self.E_sim, self.event_sim = self.update_events(S_sim, gray_img_sim)

                # update geo
                # Infer theta from events
                # events = np.transpose(np.where(self.E_real != 0))
                # sim_events = np.transpose(np.where(self.E_sim != 0))
                # self.sim.adjustment_1 = 0
                # self.sim.adjustment_2 = 0
                # self.sim.adjustment_3 = 0
                # self.sim.adjustment_4 = 0
                # diff_corrected = self.density_neignbor(self.E_real, self.E_sim)
                # # print(diff_corrected)
                # # diff_corrected = self.nearest_neighbor_mask(self.E_real, self.E_sim)
                # self.sim.adjustment_1 = self.K1 * diff_corrected
                # self.sim.adjustment_2 = self.K2 * diff_corrected
                # self.sim.adjustment_3 = self.K3 * diff_corrected
                # self.sim.adjustment_4 = self.K4 * diff_corrected





                th_from_event = self.compute_th(self.E_real)
                th_sim = self.compute_th(self.E_sim)
                self.sim.adjustment_1 = 0
                self.sim.adjustment_2 = 0
                self.sim.adjustment_3 = 0
                self.sim.adjustment_4 = 0
                events_coord = np.where(self.E_real != 0)
                diff_naive = th_sim - th_from_event
                # wrap - around effect
                if diff_naive > np.pi:
                    diff_corrected = diff_naive - (2 * np.pi)
                elif diff_naive < -np.pi:
                    diff_corrected = diff_naive + (2 * np.pi)
                else:
                    diff_corrected = diff_naive
                if 100 <= len(events_coord[0]) < 10000:
                    if not np.isnan(th_from_event) and not np.isnan(th_sim):
                        self.sim.adjustment_1 = self.K1 * diff_corrected
                        self.sim.adjustment_2 = self.K2 * diff_corrected
                        self.sim.adjustment_3 = self.K3 * diff_corrected
                        # self.sim.adjustment_4 = self.K4 * diff_corrected
                length.append(self.sim.w)

                # if self.desired_pos:
                #     d_pos.append(float(self.desired_pos))
                #     self.generate_torque()
                # else:
                #     d_pos.append(0)

                # if count >= 20:
                #
                #     # arr_length = np.array(length)
                #     # if len(arr_length) <= 7:
                #     #     self.update_length(dt)
                #     # elif np.mean((arr_length[-5:] - arr_length[-6:-1])**2) > 0.00001:
                #     #     print(np.mean((arr_length[-5:] - arr_length[-6:-1])**2))
                #     # self.update_length(dt)
                #     self.update_width(dt)
                #     count = 0
                #     print(self.sim.w)

            if self.time.time() - time2 >= 10 * self.interval:
                time2 = self.time.time()
                if self.event_sim is not None:
                    cv2.imshow("camera", self.event_real)
                    cv2.imshow("pendulum", self.event_sim)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                    # if cv2.waitKey(1) & self.flag == 1:
                        break


        # plt.plot(u, label='u')
        # plt.subplot(1, 2, 1)
        # plt.plot(u, label='u')
        # plt.plot(real_ths, label='real')
        # plt.plot(time_step, sim_ths, label='sim')
        # text_save("D:\Cam-MLMI\Visual sensing for robot estimation and control\codes\ex_results/time_step.txt", time_step)
        # text_save("D:\Cam-MLMI\Visual sensing for robot estimation and control\codes\ex_results/sim_ths.txt",
        #           sim_ths)
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(c, label='c')
        # plt.plot(sim_ths, label='sim')
        # plt.plot(time_step, real_ths, label='real')
        # plt.plot(time_step, length, label="sim_wdith")
        # plt.axhline(y=0.2, c="r", ls="--", lw=2)
        # plt.plot(time_step, u, label='c')
        # plt.show()
        # plt.legend()

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i])
        file.write(s+"\n")
    file.close()



if __name__ == "__main__":
    cam = EventCamera()
    cam.event_camera_sim()
