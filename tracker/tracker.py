import cv2
import numpy as np

import time
import warnings
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.siam_net import SiamFC
from tools.processing import get_z_image, get_x_image, get_сhange_scale_image

class SiamFCTracker:
	def __init__(self, path_weight):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = SiamFC().to(self.device)
		self.model.load_state_dict(torch.load(path_weight, map_location=self.device))
		self.model.eval()
		
		self.transform = transforms.ToTensor()

		# параметры трекинга
		self.num_scale = 3
		self.cosine_coeff = 0.176
		self.scale_step = 1.0375
		self.scale_penalty = 0.9745
		self.total_stride = 8
		
		# параметры для обработки кадров
		self.context_amount = 0.5
		self.z_image_size = 127
		self.x_image_size = 255
		self.scale_lr = 0.59

	def init_tracker(self, frame, x1, y1, w, h):
		"""
		Инициализируем вспомогательные атрибуты с помощью первого кадра
		"""
		self.bbox = (x1, y1, x1 - 1 + w, y1 - 1 + h)
		self.pos = np.array([x1 - 1 + (w - 1) / 2, y1 - 1 + (h - 1) / 2])  # координаты центра объекта
		self.target_sz = np.array([w, h]) # ширина и высота
		
		# обрабатываем изобржение с объектом интереса
		self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
		exemplar_img, scale_z, new_wh_z = get_z_image(frame, self.bbox, self.z_image_size, self.context_amount, self.img_mean)

		# получаем признаки изобржения с объектом интереса
		exemplar_img = self.transform(exemplar_img)[None, :, :, :]
		self.exemplar_img_feats = self.model.feature_extractor(exemplar_img.to(self.device))
		self.exemplar_img_feats = torch.cat([self.exemplar_img_feats for _ in range(self.num_scale)], dim=0)

		# коэффициент масштаба 
		self.penalty = np.ones((self.num_scale)) * self.scale_penalty
		self.penalty[self.num_scale // 2] = 1

		# создаем окно сглаживания косинуса
		self.interp_response_sz = 16 * 17
		self.cosine_window = np.hanning(int(self.interp_response_sz))[:, np.newaxis].dot(np.hanning(int(self.interp_response_sz))[np.newaxis, :])
		self.cosine_window = self.cosine_window.astype(np.float32)
		self.cosine_window /= np.sum(self.cosine_window)

		# масштабы для уточнее изменения размера объекта на кадре
		self.scales = self.scale_step ** np.arange(np.ceil(self.num_scale / 2) - self.num_scale, np.floor(self.num_scale / 2) + 1)

		# границы изменения (мин макс) масштаба 
		self.s_x = new_wh_z + (self.x_image_size - self.z_image_size) / scale_z
		self.min_s_x = 0.2 * self.s_x
		self.max_s_x = 5 * self.s_x


	def next_frame(self, next_frame):
		"""
		Обновление позиции объекта на следующем кадре
		"""
		# создаем изображения разного разрешения
		size_x_scales = self.s_x * self.scales
		image_scales = get_сhange_scale_image(next_frame, self.pos, self.x_image_size, size_x_scales, self.img_mean)
		
		# получаем признаки разных масштабов
		z_imgs = []
		for x in image_scales:
			z_imgs.append(self.transform(x)[None, :, :, :])
		z_imgs = torch.cat(z_imgs, dim=0)
		
		# карта корреляции 
		response_maps = self.model.get_corr(self.exemplar_img_feats, self.model.feature_extractor(z_imgs.to(self.device)))
		response_maps = response_maps.data.cpu().numpy().squeeze()
		

		response_maps_up = []
		# изменяем размер карт корреляций  
		for x in response_maps:
			response_maps_up.append(cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC))

		
		# находим макс корреляцию из разных разрешений
		max_score = np.array([x.max() for x in response_maps_up]) * self.penalty


		# выбираем макс корреляцию
		scale_idx = max_score.argmax()
		response_map = response_maps_up[scale_idx]
		# нормализация значений корреляции min-max
		response_map -= response_map.min()
		response_map /= response_map.sum()
		# сглаживание карты с помощью окна косинуса
		response_map = (1 - self.cosine_coeff) * response_map + self.cosine_coeff * self.cosine_window
	
		# находим позицию x y максимума
		max_y, max_x = np.unravel_index(response_map.argmax(), response_map.shape)
		
		# смещение относительно центра карты корреляции
		disp_response_interp = np.array([max_x, max_y]) - (self.interp_response_sz - 1) / 2.
		
		# смещение на входе
		disp_response_input = disp_response_interp * self.total_stride / 16
		
		# смещение в кадре
		scale = self.scales[scale_idx]
		disp_response_frame = disp_response_input * (self.s_x * scale) / self.x_image_size
		
		# смещаем bounding box
		self.pos += disp_response_frame
		
		# изменяем масштаб bounding box
		self.s_x *= ((1 - self.scale_lr) + self.scale_lr * scale)
		self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
		
		self.target_sz = ((1 - self.scale_lr) + self.scale_lr * scale) * self.target_sz
		
		xmin = self.pos[0] - self.target_sz[0] / 2 + 1
		ymin = self.pos[1] - self.target_sz[1] / 2 + 1
		xmax = self.pos[0] + self.target_sz[0] / 2 + 1
		ymax = self.pos[1] + self.target_sz[1] / 2 + 1		

		return (xmin, ymin, xmax, ymax)