import math
import cv2
import numpy as np


def xyxy2cxcywh(bbox):
	"""
	Преобразование (xmin ymin xmax ymax) в (x_center y_center width height) формат
	"""
	xmin, ymin, xmax, ymax = bbox
	return (xmin + xmax - 1) / 2, (ymin + ymax - 1) / 2, xmax - xmin, ymax - ymin


def crop_and_pad(image, cx, cy, model_sz, original_size, img_mean):
	"""
	Обрезаем изображение с учетом отступов
	"""
	height, width, _ = image.shape
	radius = original_size // 2

	# координаты кропа
	xmin, xmax = int(cx - radius), int(cx + radius)
	ymin, ymax = int(cy - radius), int(cy + radius)

	# учитываем границы оригинального изображения
	# и корректируем координаты кропа
	pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
	if xmin < 0:
		pad_left = -xmin
		xmin = 0
	if ymin < 0:
		pad_top = -ymin
		ymin = 0
	if xmax > width:
		pad_right = xmax - width + 1
		xmax = width
	if ymax > height:
		pad_bottom = ymax - height + 1
		ymax = height

	# кроп изображения пока без учета отступов
	crop_image = image[ymin:ymax, xmin:xmax]
	
	# добавления до нужного размера, padding средним значения пикселя
	if pad_left != 0 or pad_right != 0 or pad_bottom != 0 or pad_top != 0:
		crop_image = cv2.copyMakeBorder(crop_image, pad_top, pad_bottom, pad_left, pad_right,
				cv2.BORDER_CONSTANT, value=img_mean)

	# если размер не совпадает на всякий случай изменяем на нужный размер
	if model_sz != original_size:
		crop_image = cv2.resize(crop_image, (model_sz, model_sz))
	return crop_image


def get_z_image(image, bbox, size_z, p, img_mean):
	"""
	Преобразование кадра в изображение интереса
	"""
	cx, cy, w, h = xyxy2cxcywh(bbox)
	margin_wh = p * (w + h)
	
	wc_z = w + margin_wh
	hc_z = h + margin_wh
	
	new_wh_z = math.sqrt(wc_z * hc_z)
	scale_z = size_z / new_wh_z
	
	z_image = crop_and_pad(image, cx, cy, size_z, new_wh_z, img_mean)
	
	return z_image, scale_z, new_wh_z


def get_x_image(image, bbox, size_z, size_x, p, img_mean):
	"""
	Преобразование кадра в поисковое изображение
	"""
	cx, cy, w, h = xyxy2cxcywh(bbox)
	margin_wh = p * (w + h)

	wc_z = w + margin_wh
	hc_z = h + margin_wh
	
	new_wh_z = math.sqrt(wc_z * hc_z)
	scale_z = size_z / new_wh_z
	
	d_search = (size_x - size_z) / 2
	pad = d_search / scale_z
	new_wh_x = new_wh_z + 2 * pad
	scale_x = size_x / new_wh_x
	
	x_image = crop_and_pad(image, cx, cy, size_x, new_wh_x, img_mean)
	
	return x_image, scale_x, new_wh_x


def get_сhange_scale_image(image, center_xy, size_x, size_x_scales, img_mean):
	"""
	Создаем кадры разного разрешения, учитывая изменение масштаба интересующего объекта
	"""
	image_scaled = []
	xc, yc = center_xy
	for size_x_scale in size_x_scales:
		image_scaled.append(crop_and_pad(image, xc, yc, size_x, size_x_scale, img_mean))
	
	return image_scaled