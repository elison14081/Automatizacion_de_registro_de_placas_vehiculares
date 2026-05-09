import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from flask import current_app
import os
import re
from datetime import datetime, timedelta
import threading

class OCRService:
    def __init__(self):
        modelo_path = current_app.config.get('YOLO_MODEL_PATH', 'best.pt')
        self.modelo = YOLO(modelo_path)
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
        self.lock = threading.Lock()
        self.cap = None
        
        # ALPR Cache Buffer
        self.ultima_placa = None
        self.tiempo_ultima_placa = None
        
        # Regex peruanos
        self.regex_standard = re.compile(r'^[A-Z0-9]{3}[0-9]{3}$')
        self.regex_moto1 = re.compile(r'^[0-9]{4}[A-Z0-9]{2}$')
        self.regex_moto2 = re.compile(r'^[A-Z0-9]{2}[0-9]{4}$')
        
    def detectar_placa_desde_imagen(self, imagen_path):
        """Detecta placa desde archivo de imagen"""
        frame = cv2.imread(imagen_path)
        if frame is None:
            return None
        res = self._procesar_frame(frame)
        return res['plate_text'] if res and res['plate_text'] != 'INVALID_PLATE' else None
    
    def detectar_placa_desde_array(self, imagen_array):
        """Detecta placa desde array numpy (imagen en memoria)"""
        res = self._procesar_frame(imagen_array)
        return res['plate_text'] if res and res['plate_text'] != 'INVALID_PLATE' else None
    
    def _optimizar_imagen_placa(self, placa_roi):
        """Aplica el Image Optimization Pipeline"""
        gray = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def _validar_placa(self, texto):
        """Validación Regex para el contexto peruano AAP"""
        texto_limpio = ''.join(filter(str.isalnum, texto)).upper()
        
        if self.regex_standard.match(texto_limpio):
            return texto_limpio, "PARTICULAR_COMERCIAL"
        elif self.regex_moto1.match(texto_limpio) or self.regex_moto2.match(texto_limpio):
            return texto_limpio, "MOTO_MOTOTAXI"
            
        return "INVALID_PLATE", "UNKNOWN"
    
    def _procesar_frame(self, frame):
        """Procesa un frame y extrae la placa en formato JSON"""
        with self.lock:
            resultados = self.modelo(frame, conf=0.40, verbose=False)
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            return None
            
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                if conf < 0.40:
                    continue
                
                h, w = frame.shape[:2]
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * 0.10)
                pad_y = int(box_h * 0.10)
                
                nx1 = max(0, x1 - pad_x)
                ny1 = max(0, y1 - pad_y)
                nx2 = min(w, x2 + pad_x)
                ny2 = min(h, y2 + pad_y)
                    
                placa_roi = frame[ny1:ny2, nx1:nx2]
                
                if placa_roi.size > 0:
                    roi_optimizada = self._optimizar_imagen_placa(placa_roi)
                    resultado_ocr = self.reader.readtext(roi_optimizada)
                    
                    if not resultado_ocr:
                        resultado_ocr = self.reader.readtext(placa_roi)
                        
                    if resultado_ocr:
                        mejor_texto = max(resultado_ocr, key=lambda x: x[2])[1]
                        placa_validada, categoria = self._validar_placa(mejor_texto)
                        
                        return {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "plate_text": placa_validada,
                            "category": categoria,
                            "confidence": conf,
                            "coordinates": [nx1, ny1, nx2, ny2]
                        }
        return None
    
    def iniciar_camara(self, fuente=None):
        """Inicia la cámara para streaming"""
        if fuente is not None:
            if isinstance(fuente, str) and fuente.isdigit():
                self.fuente_camara = int(fuente)
            else:
                self.fuente_camara = fuente
        elif not hasattr(self, 'fuente_camara'):
            self.fuente_camara = 0
            
        # NUNCA usar VideoCapture para URLs de red (causa Deadlock en FFMPEG)
        if isinstance(self.fuente_camara, str) and self.fuente_camara.startswith("http"):
            return True
            
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.fuente_camara)
            if isinstance(self.fuente_camara, int):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()
    
    def cambiar_camara(self, nueva_fuente):
        """Cambia la fuente de video en tiempo real"""
        self.detener_camara()
        
        # Siempre forzar el uso de shot.jpg para evitar bloqueos de stream
        if isinstance(nueva_fuente, str) and nueva_fuente.startswith("http"):
            if "shot.jpg" not in nueva_fuente:
                if nueva_fuente.endswith("/video"):
                    nueva_fuente = nueva_fuente.replace("/video", "/shot.jpg")
                elif nueva_fuente.endswith("/"):
                    nueva_fuente += "shot.jpg"
                else:
                    nueva_fuente += "/shot.jpg"
                
        if isinstance(nueva_fuente, str) and nueva_fuente.isdigit():
            self.fuente_camara = int(nueva_fuente)
        else:
            self.fuente_camara = nueva_fuente
            
        return self.iniciar_camara()
    
    def detener_camara(self):
        """Detiene la cámara"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def obtener_frame_con_deteccion(self):
        """Obtiene un frame de la cámara con detección dibujada y actualiza el caché"""
        frame = None
        
        # Modo Cámara IP (HTTP Snapshot Bypass Total)
        if isinstance(self.fuente_camara, str) and self.fuente_camara.startswith("http"):
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(self.fuente_camara, timeout=2, verify=False, headers=headers)
                if response.status_code == 200:
                    arr = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(arr, -1)
                else:
                    print(f"[DEBUG IP CAM] Error HTTP {response.status_code} al acceder a {self.fuente_camara}")
                    return None, None
            except requests.exceptions.Timeout:
                print(f"[DEBUG IP CAM] Timeout al conectar con {self.fuente_camara}. Esperando...")
                return None, None
            except Exception as e:
                print(f"[DEBUG IP CAM] Error de red: {e}")
                return None, None
                
        # Modo Webcam Local (USB)
        else:
            if self.cap is None or not self.cap.isOpened():
                return None, None
                
            ret, frame = self.cap.read()
            if not ret:
                return None, None
                
        if frame is None:
            return None, None
            
        res = self._procesar_frame(frame)
        placa_detectada = None
        
        if res and res['plate_text'] != 'INVALID_PLATE':
            placa_detectada = res['plate_text']
            
            # GUARDAR EN EL CACHÉ ALPR SI ES VÁLIDA
            self.ultima_placa = res
            self.tiempo_ultima_placa = datetime.now()
            
            x1, y1, x2, y2 = res['coordinates']
            conf = res['confidence']
            cat = res['category']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"PLACA: {placa_detectada} ({cat})", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        return frame, placa_detectada
    
    def capturar_placa(self):
        """Retorna la última placa validada del caché o captura una nueva (JSON)"""
        # Si tenemos una placa válida en el caché de los últimos 3 segundos, la usamos
        if self.ultima_placa and self.tiempo_ultima_placa:
            if datetime.now() - self.tiempo_ultima_placa < timedelta(seconds=3):
                res_cache = self.ultima_placa
                # Limpiamos el caché para no enviar duplicados si se queda parqueado
                self.ultima_placa = None 
                return res_cache
                
        # Si el caché está vacío o caducó, intentamos capturar un frame nuevo
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return self._procesar_frame(frame)