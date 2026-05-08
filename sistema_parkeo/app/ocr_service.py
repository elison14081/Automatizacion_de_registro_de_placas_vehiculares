import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from flask import current_app
import os

class OCRService:
    def __init__(self):
        modelo_path = current_app.config.get('YOLO_MODEL_PATH', 'best.pt')
        self.modelo = YOLO(modelo_path)
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
        self.cap = None
        
    def detectar_placa_desde_imagen(self, imagen_path):
        """Detecta placa desde archivo de imagen"""
        frame = cv2.imread(imagen_path)
        if frame is None:
            return None
            
        return self._procesar_frame(frame)
    
    def detectar_placa_desde_array(self, imagen_array):
        """Detecta placa desde array numpy (imagen en memoria)"""
        return self._procesar_frame(imagen_array)
    
    def _procesar_frame(self, frame):
        """Procesa un frame y extrae la placa"""
        resultados = self.modelo(frame)
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            return None
            
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Solo procesar si la confianza es alta
                if conf < 0.5:
                    continue
                    
                placa_roi = frame[y1:y2, x1:x2]
                
                if placa_roi.size > 0:
                    resultado_ocr = self.reader.readtext(placa_roi)
                    if resultado_ocr:
                        placa_texto = resultado_ocr[0][1]
                        placa_texto = ''.join(filter(str.isalnum, placa_texto)).upper()
                        
                        # Validar que tenga formato mínimo de placa
                        if len(placa_texto) >= 5:
                            return placa_texto
        
        return None
    
    def iniciar_camara(self):
        """Inicia la cámara para streaming"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()
    
    def detener_camara(self):
        """Detiene la cámara"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def obtener_frame_con_deteccion(self):
        """Obtiene un frame de la cámara con detección dibujada"""
        if self.cap is None or not self.cap.isOpened():
            return None, None
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        resultados = self.modelo(frame)
        placa_detectada = None
        
        # Dibujar detecciones
        if len(resultados) > 0 and len(resultados[0].boxes) > 0:
            for r in resultados:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    label = r.names[int(box.cls[0])]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Intentar OCR
                    placa_roi = frame[y1:y2, x1:x2]
                    if placa_roi.size > 0:
                        resultado_ocr = self.reader.readtext(placa_roi)
                        if resultado_ocr:
                            placa_texto = resultado_ocr[0][1]
                            placa_texto = ''.join(filter(str.isalnum, placa_texto)).upper()
                            placa_detectada = placa_texto
                            cv2.putText(frame, f"PLACA: {placa_texto}", (x1, y2+30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        return frame, placa_detectada
    
    def capturar_placa(self):
        """Captura una placa de la cámara actual"""
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return self._procesar_frame(frame)