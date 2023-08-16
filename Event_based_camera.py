import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.neighbors import KDTree
from pendulum import PendulumEnv
import serial
import threading
import asyncio


class EventCamera:
    def __init__(self, camera_id, fps=90):
        self.desired_pos = None
        self.sim = PendulumEnv(render_mode='rgb_array', l=1.5, w=0.2)
        self.act = PendulumEnv(render_mode='rgb_array', l=1.5, g=2)
        self.act.c = 0.4
        self.event_sim = None
        self.event_real = None
        self.E_real = None
        self.E_sim = None
        self.interval = 0.001
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        # self.cap = cv2.VideoCapture("D:\Cam-MLMI\Visual sensing for robot estimation and control\codes/video_output_l=10.avi")
        self.time = time
        self.flag = 0

        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(5, fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.offset = np.array([self.sim.screen_h // 2, -self.sim.screen_w // 2])
        # self.out = cv2.VideoWriter('video_output.avi', int(self.cap.get(cv2.CAP_PROP_FOURCC)), 40.0,
        #                            (self.width, self.height))

        self.K1 = - 50
        self.K2 = - 400
        self.K3 = - 1000
        self.K4 = - 0.1
        self.K5 = - 0.1

        self.window = np.ones([9, 9])

    def receiver(self, ser):
        # ser.flushInput()
        ser.write(1)
        while ser.in_waiting == 0:
            pass
        data = ser.read_all()
        # print(data)
        value = data.decode('utf-8')

        real_th = np.deg2rad(float(value))
        real_th = real_th % (2 * np.pi)
        if real_th > np.pi:
            real_th -= 2.0 * np.pi
        return real_th

    def send_torque(self):
        # positional
        # u = float(self.desired_pos) - self.sim.state[0] + 0.01 * self.error #+ 0.014 * self.de
        # if u > np.pi:
        #     u = u - (2 * np.pi)
        # elif u < -np.pi:
        #     u = u + (2 * np.pi)
        # else:
        #     u = u
        # u += np.pi

        # pose and velo
        u = 0.58 * float(self.act.state[0]) - self.sim.state[0] + 0.04 * self.error[0] + 0.000001 * self.de_p

        v = 0.7 * float(self.act.state[1]) - self.sim.state[1] + 0.05 * self.error[1]+ 0.000002 * self.de_v
        u = 0.8 * u + 0.02 * v
        if u > np.pi:
            u = u - (2 * np.pi)
        elif u < -np.pi:
            u = u + (2 * np.pi)
        else:
            u = u
        # t = max(t, -2)
        # t = max(min(0.8 * u + 0.02 * v, 3),-3) + 3
        u += np.pi
        ser.write(int(40 * u))

    async def read_user_input(self):
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input)
            if user_input != 'q':
                # Invoke the callback function with the user input as an argument
                self.desired_pos = user_input
                self.new_time = time.time()
            else:
                self.flag = 1

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

        outer_radii = np.arange(80, max_radius + 1, ring_width)
        inner_radii = outer_radii - ring_width

        y_indices, x_indices = np.indices((height, width))

        distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

        masks = []

        for inner_radius, outer_radius in zip(inner_radii, outer_radii):
            mask = np.logical_and(inner_radius <= distances, distances < outer_radius).astype(np.uint8)
            masks.append((mask, outer_radius - ring_width // 2))

        return masks

    def update_events(self, S, gray_img, c=0.5, filter=False):
        c1 = 2 ** c
        c2 = 2 ** (-c)
        E_pos = ((gray_img + 25) / (S + 25) > c1) + 0
        E_neg = ((gray_img + 25) / (S + 25) < c2) + 0

        if filter:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
                cv2.circle(self.event_real, (320, 180), int(r - self.ring_width // 2), (0, 0, 255))
                cv2.circle(self.event_sim, (320, 180), int(r - self.ring_width // 2), (0, 0, 255))
                cv2.circle(self.event_sim, (int(sim_center_y), int(sim_center_x)), 5,
                           (128, 0, 128), -1)
                cv2.circle(self.event_real, (int(real_center_y), int(real_center_x)), 5,
                           (0, 64, 128), -1)
                simx = simx + sim_center_x / num_of_points
                simy = simy + sim_center_y / num_of_points
                vecx = vecx + (sim_center_x - real_center_x) / (num_of_points * r)
                vecy = vecy + (sim_center_y - real_center_y) / (num_of_points * r)

            cv2.circle(self.event_real, (320, 180), int(self.masks[-1][1] + self.ring_width // 2), (0, 0, 255))
            cv2.circle(self.event_sim, (320, 180), int(self.masks[-1][1] + self.ring_width // 2), (0, 0, 255))

            simx = (simx - center_x) / self.sim.screen_w
            simy = (simy - center_y) / self.sim.screen_h

        diff_corrected = (-simy * vecx + simx * vecy)
        return diff_corrected

    def nearest_neighbor(self, events, sim_events):
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
                distance, index = tree.query(point, k=1, return_distance=True)
                index = index[0][0]
                # cv2.circle(self.event_real, (int(tree.data[index][1]), int(tree.data[index][0])), 5, (0, 0, 255), -1)
                # cv2.circle(self.event_sim, (b, a), 5, (0, 0, 255), -1)

                simx = simx + point[0][0] / 50
                simy = simy + point[0][1] / 50
                vecx = vecx + (point[0][0] - tree.data[index][0]) / 50
                vecy = vecy + (point[0][1] - tree.data[index][1]) / 50

            simx -= 0.5
            simy -= 0.5

            diff_corrected = - simy * vecx + simx * vecy
            # diff_corrected = np.sign(vecy) * ((vecy ** 2 + vecx ** 2) ** 0.5)
        else:
            diff_corrected = 0

        return diff_corrected

    def density_neignbor(self, E_real, E_sim):

        if self.E_real.any() != 0 and self.E_sim.any() != 0:

            non_zero_counts_real = cv2.filter2D(E_real, ddepth=-1, kernel=self.window)
            # num = 3
            #
            # simx = 0
            # simy = 0
            # realx = 0
            # realy = 0
            a = np.max(non_zero_counts_real)
            b = np.min(non_zero_counts_real)
            # for i in range(num):
            #     weight = (i + 1) // num
            #     index1 = np.mean(np.where(non_zero_counts_real == a * weight), axis=1) / [self.sim.screen_w,
            #                                                                                self.sim.screen_h]
            #     index2 = np.mean(np.where(non_zero_counts_real == b * weight), axis=1) / [self.sim.screen_w,
            #                                                                              self.sim.screen_h]
            #     realx = realx + (index1[0] + index2[0]) * weight
            #     realy = realy + (index2[1] + index2[1]) * weight

            first_index = np.mean(np.where(non_zero_counts_real == a), axis=1) / [self.sim.screen_w, self.sim.screen_h]
            second_index = np.mean(np.where(non_zero_counts_real == a // 2), axis=1) / [self.sim.screen_w,
                                                                                        self.sim.screen_h]

            third_index = np.mean(np.where(non_zero_counts_real == b), axis=1) / [self.sim.screen_w, self.sim.screen_h]
            forth_index = np.mean(np.where(non_zero_counts_real == b // 2), axis=1) / [self.sim.screen_w,
                                                                                       self.sim.screen_h]

            non_zero_counts_sim = cv2.filter2D(E_sim, ddepth=-1, kernel=self.window)

            a_sim = np.max(non_zero_counts_sim)
            b_sim = np.min(non_zero_counts_sim)
            # for i in range(num):
            #     weight = (i + 1) // num
            #     index1 = np.mean(np.where(non_zero_counts_sim == a_sim * weight), axis=1) / [self.sim.screen_w,
            #                                                                               self.sim.screen_h]
            #     index2 = np.mean(np.where(non_zero_counts_sim == b_sim * weight), axis=1) / [self.sim.screen_w,
            #                                                                               self.sim.screen_h]
            #     simx = simx + (index1[0] + index2[0]) * weight
            #     simy = simy + (index2[1] + index2[1]) * weight

            first_index_sim = np.mean(np.where(non_zero_counts_sim == a_sim), axis=1) / [self.sim.screen_w,
                                                                                         self.sim.screen_h]
            second_index_sim = np.mean(np.where(non_zero_counts_sim == a_sim // 2), axis=1) / [self.sim.screen_w,
                                                                                               self.sim.screen_h]

            b_sim = np.min(non_zero_counts_sim)
            third_index_sim = np.mean(np.where(non_zero_counts_sim == b_sim), axis=1) / [self.sim.screen_w,
                                                                                         self.sim.screen_h]
            forth_index_sim = np.mean(np.where(non_zero_counts_sim == b_sim // 2), axis=1) / [self.sim.screen_w,
                                                                                              self.sim.screen_h]

            simx = first_index_sim[0] / 3 + second_index_sim[0] / 6 + third_index_sim[0] / 3 + forth_index_sim[0] / 6
            simy = first_index_sim[1] / 3 + second_index_sim[1] / 6 + third_index_sim[1] / 3 + forth_index_sim[1] / 6
            realx = first_index[0] / 3 + second_index[0] / 6 + third_index[0] / 3 + forth_index[0] / 6
            realy = first_index[1] / 3 + second_index[1] / 6 + third_index[1] / 3 + forth_index[1] / 6
            if not np.isnan(simy) and not np.isnan(simx) and not np.isnan(realy) and not np.isnan(realx):
                # cv2.circle(self.event_sim, (int(self.sim.screen_h * simy), int(self.sim.screen_w * simx)), 5, (255, 0, 0), -1)
                # cv2.circle(self.event_real, (int(self.sim.screen_h * realy), int(self.sim.screen_w * realx)), 5, (255, 0, 0), -1)
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

    def nearest_neighbor_in_radius(self, E_real, E_sim):
        num_of_points = 50
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
            for i in range(num_of_points // 2):
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
                for i in range(num_of_points // 2):
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

    def draw_center(self, E, event):
        events_coord = np.where(E != 0)
        if len(events_coord[0]) >= 100:
            center_x, center_y = np.mean(np.where(E != 0), axis=1)
            cv2.circle(event, (int(center_y), int(center_x)), 10, (0, 0, 255), -1)

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
            for i in range(num_of_points // 2):
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
                for j in range(num_of_points // 2):
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

    def event_camera_sim(self):
        # self.occlusion = self.creat_rectangle_mask()
        f, img = self.cap.read()
        S_real = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(int)
        # S_real = S_real * self.occlusion[0]
        self.act.reset()
        self.act.state = [-0.6 * np.pi, 0.0]

        self.sim.reset()
        S_sim = cv2.cvtColor(np.squeeze(self.sim.render()), cv2.COLOR_BGR2GRAY).astype(int)
        time1 = self.time.time()
        time2 = self.time.time()
        sim_ths = []
        start = time.time()
        time_step = []
        process_time = []
        real_ths = []
        frame_counter = 0
        d_pos = []
        error = []
        new_time = []
        self.error = [0, 0]
        self.ring_width = 20
        self.masks = self.creat_ring_masks(S_real, self.ring_width)
        count = 0
        pre_img = None
        self.pre_error = [0, 0]

        self.input_thread = threading.Thread(target=asyncio.run, args=(self.read_user_input(),))
        self.input_thread.start()

        while self.cap.isOpened():

            if self.time.time() - time1 >= 0.02:
                dt = self.time.time() - time1
                time1 = time.time()
                # observation, reward, done, info = self.sim.step([0], dt, True)
                observation, reward, done, info = self.act.step([0], dt, True)

                time_step.append(time.time() - start)
                sim_ths.append(self.sim.state[0])
                # real_th = self.receiver(ser)
                # if real_th is None:
                #     del time_step[-1]
                #     del sim_ths[-1]
                # else:
                #     real_ths.append(real_th)
                # frame_counter += 1
                # if frame_counter == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                #     frame_counter = 0
                #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                f, img = self.cap.read()
                gray_img_real = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(int)
                # gray_img_real = gray_img_real * self.occlusion[0]
                S_real, self.E_real, self.event_real = self.update_events(S_real, gray_img_real, filter=True)
                # S_real, self.E_real_un, self.event_real_unfilter = self.update_events(s, gray_img_real, filter=False)

                events_coord = np.where(self.E_real != 0)
                if len(events_coord[0]) >= 100:
                    observation, reward, done, info = self.sim.step([0], dt, True)

                    # for synthetic pendulum
                    gray_img_sim = cv2.cvtColor(np.squeeze(self.sim.render()), cv2.COLOR_BGR2GRAY).astype(int)
                    S_sim, self.E_sim, self.event_sim = self.update_events(S_sim, gray_img_sim, filter=False)

                    # self.sim.adjustment_1 = 0
                    # self.sim.adjustment_2 = 0
                    # self.sim.adjustment_3 = 0
                    # self.sim.adjustment_4 = 0
                    #
                    # time3 = time.time()
                    # # diff_corrected = self.density_neignbor(self.E_real, self.E_sim)
                    # diff_corrected = self.nearest_neighbor_mask(self.E_real, self.E_sim)
                    # # print(diff_corrected)
                    # process_time.append(time.time() - time3)
                    #
                    # self.sim.adjustment_1 = self.K1 * diff_corrected
                    # self.sim.adjustment_2 = self.K2 * diff_corrected
                    # self.sim.adjustment_3 = self.K3 * diff_corrected

                    '''update due to events'''
                    time3 = time.time()
                    th_from_event = self.compute_th(self.E_real)
                    th_sim = self.compute_th(self.E_sim)
                    self.sim.adjustment_1 = 0
                    self.sim.adjustment_2 = 0
                    self.sim.adjustment_3 = 0
                    self.sim.adjustment_4 = 0
                    events_coord = np.where(self.E_real != 0)
                    diff_naive = th_sim - th_from_event
                    if diff_naive > np.pi:
                        diff_corrected = diff_naive - (2 * np.pi)
                    elif diff_naive < -np.pi:
                        diff_corrected = diff_naive + (2 * np.pi)
                    else:
                        diff_corrected = diff_naive
                    process_time.append(time.time() - time3)
                    if 100 <= len(events_coord[0]) < 10000:
                        if not np.isnan(th_from_event) and not np.isnan(th_sim):
                            self.sim.adjustment_1 = self.K1 * diff_corrected
                            self.sim.adjustment_2 = self.K2 * diff_corrected
                            self.sim.adjustment_3 = self.K3 * diff_corrected
                else:
                    observation, reward, done, info = self.sim.step([0], 0, True)
                    gray_img_sim = cv2.cvtColor(np.squeeze(self.sim.render()), cv2.COLOR_BGR2GRAY).astype(int)
                    S_sim, self.E_sim, self.event_sim = self.update_events(S_sim, gray_img_sim, filter=False)

                if self.desired_pos:
                    # d_pos.append(float(self.desired_pos))
                    d_pos.append(float(self.act.state[0]))
                    # self.de = (float(self.desired_pos) - self.sim.state[0] - self.pre_error)/dt
                    # self.error += float(self.desired_pos) - self.sim.state[0]
                    # self.send_torque()
                    # error.append(self.sim.state[0]-float(self.desired_pos))
                    # new_time.append(time.time()-self.new_time)
                    # self.pre_error = float(self.desired_pos) - self.sim.state[0]
                    self.de_p = (self.act.state[0] - self.sim.state[0] - self.pre_error[0]) / dt
                    self.de_v = (self.act.state[1] - self.sim.state[1] - self.pre_error[1]) / dt
                    self.error[0] += self.act.state[0] - self.sim.state[0]
                    self.error[1] += self.act.state[1] - self.sim.state[1]
                    self.send_torque()
                    error.append(self.sim.state[0] - float(self.desired_pos))
                    new_time.append(time.time() - self.new_time)
                    self.pre_error[0] = float(self.desired_pos) - self.sim.state[0]

                else:
                    d_pos.append(0)

                # count += 1
                # if count >= 20:
                #     # arr_length = np.array(length)
                #     # if len(arr_length) <= 7:
                #     #     self.update_length(dt)
                #     # elif np.mean((arr_length[-5:] - arr_length[-6:-1])**2) > 0.00001:
                #     #     print(np.mean((arr_length[-5:] - arr_length[-6:-1])**2))
                #     self.update_length(dt)
                #     self.update_width(dt)
                #     count = 0

            if self.time.time() - time2 >= 10 * self.interval:
                time2 = time.time()
                # cv2.circle(img, (int(self.width // 2), int(self.height // 2)), 10, (0, 0, 255), -1)
                # cv2.line(img, (int(self.width // 2) - 100, int(self.height // 2)),
                #          (int(self.width // 2) + 100, int(self.height // 2)), (0, 0, 255))
                # cv2.line(img, (int(self.width // 2), int(self.height // 2) - 100),
                #          (int(self.width // 2), int(self.height // 2) + 100), (0, 0, 255))
                if self.event_sim is not None:
                    cv2.imshow("real", img)
                    # self.out.write(img)
                    # self.draw_center(self.E_real, self.event_real)
                    # self.draw_center(self.E_sim, self.event_sim)
                    # cv2.imwrite("img1.jpg", pre_img)
                    # cv2.imwrite("method4_real.jpg",self.event_real)
                    # cv2.imwrite("method4_sim.jpg", self.event_sim)
                    # np.save("real_events.npy", self.E_real_un)
                    cv2.imshow("camera", self.event_real)
                    cv2.imshow("pendulum", self.event_sim)
                    if cv2.waitKey(1) & self.flag == 1:
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # plt.subplot(3, 1, 1)
        # plt.plot(u, label='u', c='g')
        # plt.subplot(3, 1, 2)
        # plt.plot(sim_ths, label='theta', c='b')
        # plt.subplot(3, 1, 3)
        # plt.ylim([-2,2])
        # sim_ths = np.array(sim_ths)
        # real_ths = np.array(real_ths)
        # time_step = np.array(time_step)
        plt.plot(time_step, sim_ths, label='motor')
        plt.plot(time_step, d_pos, label='target')
        # plt.plot(time_step, real_ths, label='real')
        # text_save("D:\Cam-MLMI\Visual sensing for robot estimation and control\codes\ex_results\different_speed/time_step_speed=1.5_theta.txt",
        #           time_step)
        # text_save("D:\Cam-MLMI\Visual sensing for robot estimation and control\codes\ex_results\different_speed/sim_ths_speed=1.5_theta.txt",
        #           sim_ths)
        plt.show()
        # converge = (sim_ths[0:100] - real_ths[0:100]) / real_ths[0:100]
        # print(time_step[np.where(abs(converge) <= 0.05)], 1)
        # print(np.average(process_time), 2)
        # y = sim_ths - real_ths
        # error = []
        # for i in y[200:-1]:
        #     if i > np.pi:
        #         i = i - (2 * np.pi)
        #     elif i < -np.pi:
        #         i = i + (2 * np.pi)
        #     else:
        #         i = i
        #     error.append(i)
        # error = np.array(error)
        # print(np.mean(error ** 2), 3)
        error = np.array(error)
        d_pos = np.array(d_pos)[len(d_pos) - len(error):]
        index = np.where(abs(error / d_pos) <= 0.05)[0][0]
        print(new_time[index])
        print(np.abs(np.mean(error[-100:])))
        print(np.max(error[index + 1:]))

        self.cap.release()
        cv2.destroyAllWindows()


def text_save(filename, data):
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i])
        file.write(s + "\n")
    file.close()


if __name__ == "__main__":
    cam = EventCamera(1, 100)

    portx = "COM7"
    bps = 9600
    ser = serial.Serial(portx, int(bps), timeout=0.01, parity=serial.PARITY_NONE, stopbits=1)
    if not ser.is_open:
        ser.open()
    ser.flushInput()

    cam.event_camera_sim()

# events_coord = np.where(self.E_real != 0)
# if 100 <= len(events_coord[0]) < 10000:
#     E_real = (np.mean(events_coord, axis=1) + [1, 1]) / [self.sim.screen_w,
#                                                                    self.sim.screen_h]
#     E_sim = (np.mean(np.where(self.E_sim != 0), axis=1) + [1, 1]) / [self.sim.screen_w, self.sim.screen_h]
#
#     # Pos_change_real = np.linalg.norm(E_real - pre_E_real)
#     Pos_diff = (E_sim - E_real)[1]
#     if not np.isnan(Pos_diff):
#         self.sim.adjustment_1 = self.K1 * Pos_diff
#         self.sim.adjustment_2 = self.K2 * Pos_diff
#         self.sim.adjustment_3 = self.K3 * Pos_diff
