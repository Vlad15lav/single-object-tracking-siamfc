import argparse
import os
import glob
import cv2
import numpy as np

from tracker.tracker import SiamFCTracker


def get_args():
	parser = argparse.ArgumentParser('Video Tracking')
	parser.add_argument('--video-path', type=str, default='example/VID_20230513_134718.mp4', help='path video')
	parser.add_argument('--weight-path', type=str, default="weights/SiamFC.pth", help='path weights training')
	parser.add_argument('--scale', type=float, default=1, help='scale window size')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	opt = get_args()


	if os.path.isdir(opt.video_path):
		# tracking видео из папки с картинками		
		filenames = sorted(glob.glob(os.path.join(opt.video_path, "img/*.jpg")), key=lambda x: int(os.path.basename(x).split('.')[0]))
		path_bboxes = os.path.join(opt.video_path, "groundtruth_rect.txt")

		# преобразуем кадры в RGB
		frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]

		# открываем файл с разметкой боксов
		with open(path_bboxes) as f:
			lines_target_text = f.readlines()
		target_bboxes = [list(map(int, line.strip().split(','))) for line in lines_target_text]

		# модель SimaFC с обработкой tracking
		tracker = SiamFCTracker(opt.weight_path)

		title = opt.video_path.split('/')[-1]
		for idx, frame in enumerate(frames):
			if idx == 0: # первый кадр использует начальную разметку 
				bbox = target_bboxes[idx]
				tracker.init_tracker(frame, *bbox)
				bbox = (bbox[0] - 1, bbox[1] - 1, bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1) # x1 y1 x2 y2 
			else:
				# обновляем расположение трекинга следующего кадра
				bbox = tracker.next_frame(frame)
			
			# рисуем прямоугольник 
			frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
			
			# сравниваем с разметкой прямоугольника
			target_bbox = target_bboxes[idx]
			# x1 y1 x2 y2 
			target_bbox = (target_bbox[0], target_bbox[1], target_bbox[0] + target_bbox[2], target_bbox[1] + target_bbox[3])

			frame = cv2.rectangle(frame, (int(target_bbox[0] - 1), int(target_bbox[1] - 1)), (int(target_bbox[2]-1), int(target_bbox[3]-1)), (255, 0, 0), 1)
			
			if len(frame.shape) == 3:
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			
			frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
			
			if opt.scale > 1:
				dim = frame.shape[:2]

				frame = cv2.resize(frame, (dim[1] * opt.scale, dim[0] * opt.scale), interpolation=cv2.INTER_CUBIC)

			cv2.imshow(title, frame)
			cv2.waitKey(30)

			if cv2.waitKey(1) == ord('q'):
				exit()

	else:
		# открываем видео
		cap = cv2.VideoCapture(opt.video_path)

		if not cap.isOpened():
			print("Видео не запустилось!")
			exit()

		# модель SimaFC с обработкой tracking
		tracker = SiamFCTracker(opt.weight_path)

		# получаем первый кадр, где будет запрашиваться выделения объекта
		ret, frame = cap.read()

		# создаем функцию для события выделения объекта
		x1, y1, x2, y2 = -1, -1, -1, -1
		select_object = None
		def select_image_object(event, x, y, flags, params):
			# сохраняем координаты xmin ymin xmax ymax
			global x1, y1, x2, y2, select_object

			# мышка нажата
			if event == cv2.EVENT_LBUTTONDOWN:
				x1, y1 = x, y
			# мышка отжата
			elif event == cv2.EVENT_LBUTTONUP:
				x2, y2 = x, y
				if x1 > x:
					x1, x2 = x, x1
				if y1 > y:
					y2, y1 = y1, y

				# показываем объект интереса
				select_object = frame[y1:y2, x1:x2]
				cv2.destroyWindow("Select Object")

		# показываем первый кадр и просим выделить на нем объект
		cv2.imshow('Select Object', frame)
		cv2.setMouseCallback('Select Object', select_image_object)
		cv2.waitKey(0)

		if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
			print("Объект не был выделен!")
			exit()

		cv2.imshow("Image Tracking", select_object)
		tracker.init_tracker(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), x1, y1, x2 - x1, y2 - y1)
		bbox = (x1, y1, x2, y2)

		i_frame = 1
		while(cap.isOpened()):
			ret, frame = cap.read()
			
			if not ret:
				exit()

			bbox = tracker.next_frame(frame)
			i_frame += 1
			
			# рисуем прямоугольник трекинга
			frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

			if opt.scale != 1: # изменяем размер окна
				dim = frame.shape[:2]

				frame = cv2.resize(frame, (int(dim[1] * opt.scale), int(dim[0] * opt.scale)), interpolation=cv2.INTER_CUBIC)

			frame = cv2.putText(frame, str(i_frame), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

			cv2.imshow("Tracking", frame)
			
			if cv2.waitKey(1) == ord('q'):
				exit()

		cap.release()
		cv2.destroyAllWindows()