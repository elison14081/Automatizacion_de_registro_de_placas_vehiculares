import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from flask import current_app
import os
import re
from datetime import datetime, timedelta
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# CENTINELA VEHICULAR — Motor ALPR Perú v2.0
# Pipeline: YOLO Detection → Geometric Validation → Multi-Pipeline OCR
#           → FE-Schrift Correction → Peruvian Format Validation
# ═══════════════════════════════════════════════════════════════════════════════

class OCRService:
    def __init__(self):
        modelo_path = current_app.config.get('YOLO_MODEL_PATH', 'best.pt')
        self.modelo = YOLO(modelo_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.lock = threading.Lock()
        self.cap = None
        self.fuente_camara = None
        
        # ── ALPR Cache Buffer ──
        self.ultima_placa = None
        self.tiempo_ultima_placa = None
        
        # ── Parámetros YOLO optimizados para ALPR ──
        self.YOLO_CONF = 0.35       # Umbral de confianza (más bajo = más sensible)
        self.YOLO_IOU = 0.45        # NMS para tráfico denso
        self.YOLO_IMGSZ = 640       # Resolución de entrada (640 para CPU, 1280 para GPU)
        
        # ── Geometría de Placas Peruanas (ratios de aspecto) ──
        # Auto/Camioneta: 340mm x 185mm → ratio ~1.84
        # Moto/Mototaxi: 190mm x 110mm → ratio ~1.73
        self.RATIO_AUTO_MIN = 1.4
        self.RATIO_AUTO_MAX = 3.5
        self.RATIO_MOTO_MIN = 1.0
        self.RATIO_MOTO_MAX = 2.5
        self.MIN_PLATE_PIXELS = 40   # Ancho mínimo en px para intentar OCR
        
        # ── Regex de Formatos Peruanos (AAP/MTC) ──
        # Particulares/Comerciales: ABC-123 (3 letras + 3 números)
        self.regex_standard = re.compile(r'^[A-Z]{3}[0-9]{3}$')
        # Variante alfanumérica: A1B-234, ABC-1D3 (mixto permitido)
        self.regex_alfanum = re.compile(r'^[A-Z0-9]{3}[0-9]{3}$')
        # Motos formato 1: 1234-AB
        self.regex_moto1 = re.compile(r'^[0-9]{4}[A-Z]{2}$')
        # Motos formato 2: AB-1234
        self.regex_moto2 = re.compile(r'^[A-Z]{2}[0-9]{4}$')
        # Motos formato 3: 1234-5A (numérico + alfanumérico)
        self.regex_moto3 = re.compile(r'^[0-9]{4}[A-Z0-9]{2}$')
        # Diplomáticos/Especiales: CD-1234
        self.regex_diplomatico = re.compile(r'^[A-Z]{2}[0-9]{4}$')
        
        # ── Tabla de Corrección FE-Schrift ──
        # La tipografía FE-Schrift tiene trazos específicos para evitar
        # confusiones, pero el OCR común las confunde de todos modos.
        # Posición 0-2 (letras), Posición 3-5 (números)
        self.FE_LETRA_A_NUMERO = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1', 'T': '1',
            'Z': '2',
            'S': '5', 'J': '3',
            'B': '8', 'G': '6',
            'A': '4',
        }
        self.FE_NUMERO_A_LETRA = {
            '0': 'O', '1': 'I', '2': 'Z',
            '5': 'S', '8': 'B', '6': 'G',
            '4': 'A', '3': 'J',
        }
        
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
    
    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: Pre-procesamiento de Imagen para Material Retroreflectivo
    # ═══════════════════════════════════════════════════════════════════
    
    def _pipeline_clahe(self, roi):
        """Pipeline 1: CLAHE — Ecualización adaptativa para brillo del aluminio"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    def _pipeline_bilateral(self, roi):
        """Pipeline 2: Bilateral — Suavizado preservando bordes del filete negro"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        return cv2.adaptiveThreshold(
            bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    def _pipeline_otsu(self, roi):
        """Pipeline 3: Otsu — Umbralización global para alto contraste"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _pipeline_morph(self, roi):
        """Pipeline 4: Morfológico — Limpieza agresiva para glare retroreflectivo"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Compensar brillo extremo del material retroreflectivo
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(norm)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        # Operación morfológica para limpiar ruido del holograma
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: Corrección FE-Schrift y Validación de Formato
    # ═══════════════════════════════════════════════════════════════════
    
    def _corregir_fe_schrift(self, texto_raw):
        """Aplica corrección de caracteres FE-Schrift al texto OCR.
        
        Formato peruano estándar: LLL-NNN (3 letras + guión + 3 números)
        Las posiciones 0-2 deben ser LETRAS, las posiciones 3-5 deben ser NÚMEROS.
        """
        texto = ''.join(filter(str.isalnum, texto_raw)).upper()
        
        if len(texto) < 5:
            return texto
            
        # Si tiene exactamente 6 caracteres, aplicar corrección posicional
        if len(texto) == 6:
            corregido = list(texto)
            
            # Posiciones 0-2: deberían ser LETRAS
            for i in range(3):
                if corregido[i].isdigit():
                    corregido[i] = self.FE_NUMERO_A_LETRA.get(corregido[i], corregido[i])
            
            # Posiciones 3-5: deberían ser NÚMEROS
            for i in range(3, 6):
                if corregido[i].isalpha():
                    corregido[i] = self.FE_LETRA_A_NUMERO.get(corregido[i], corregido[i])
            
            return ''.join(corregido)
        
        return texto
    
    def _validar_placa(self, texto):
        """Validación y clasificación de placa peruana con corrección FE-Schrift."""
        texto_limpio = ''.join(filter(str.isalnum, texto)).upper()
        
        if len(texto_limpio) < 4:
            return "INVALID_PLATE", "UNKNOWN"
        
        # Intentar corrección FE-Schrift
        texto_corregido = self._corregir_fe_schrift(texto_limpio)
        
        # Clasificar formato
        if self.regex_standard.match(texto_corregido):
            return texto_corregido, "PARTICULAR"
        elif self.regex_alfanum.match(texto_corregido):
            return texto_corregido, "COMERCIAL"
        elif self.regex_moto1.match(texto_corregido) or self.regex_moto3.match(texto_corregido):
            return texto_corregido, "MOTO"
        elif self.regex_moto2.match(texto_corregido) or self.regex_diplomatico.match(texto_corregido):
            return texto_corregido, "ESPECIAL"
        
        # Si la corrección no matcheó, probar con el texto original
        if self.regex_standard.match(texto_limpio) or self.regex_alfanum.match(texto_limpio):
            return texto_limpio, "PARTICULAR"
        elif self.regex_moto1.match(texto_limpio) or self.regex_moto2.match(texto_limpio):
            return texto_limpio, "MOTO"
        
        # Aceptar texto si tiene longitud razonable (para pruebas)
        if len(texto_corregido) >= 5:
            return texto_corregido, "DETECTADO"
            
        return "INVALID_PLATE", "UNKNOWN"
    
    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: Validación Geométrica de Detección
    # ═══════════════════════════════════════════════════════════════════
    
    def _validar_geometria(self, box_w, box_h):
        """Valida si las proporciones del rectángulo coinciden con una placa real.
        
        Placas peruanas:
          - Auto: 340 x 185 mm → ratio 1.84 (rango: 1.4 - 3.5)
          - Moto: 190 x 110 mm → ratio 1.73 (rango: 1.0 - 2.5)
        
        Rechaza: logos cuadrados, stickers del parabrisas muy alargados, señales de tránsito.
        """
        if box_h == 0 or box_w < self.MIN_PLATE_PIXELS:
            return False, "UNKNOWN"
            
        ratio = box_w / box_h
        
        if self.RATIO_AUTO_MIN <= ratio <= self.RATIO_AUTO_MAX:
            return True, "AUTO"
        elif self.RATIO_MOTO_MIN <= ratio <= self.RATIO_MOTO_MAX:
            return True, "MOTO"
        
        return False, "UNKNOWN"
    
    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4: Procesamiento Principal — YOLO + Multi-Pipeline OCR
    # ═══════════════════════════════════════════════════════════════════
    
    def _procesar_frame(self, frame):
        """Pipeline principal: YOLO Detection → Geometry → OCR → Validation"""
        with self.lock:
            resultados = self.modelo(
                frame,
                conf=self.YOLO_CONF,
                iou=self.YOLO_IOU,
                imgsz=self.YOLO_IMGSZ,
                verbose=False
            )
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            return None
        
        mejor_resultado = None
        mejor_confianza = 0
        
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                h, w = frame.shape[:2]
                box_w = x2 - x1
                box_h = y2 - y1
                
                # ── STAGE 3: Validación geométrica ──
                es_placa, tipo_vehiculo = self._validar_geometria(box_w, box_h)
                if not es_placa:
                    continue
                
                # Padding adaptativo según tipo de vehículo
                pad_x = int(box_w * 0.08)
                pad_y = int(box_h * 0.12)
                
                nx1 = max(0, x1 - pad_x)
                ny1 = max(0, y1 - pad_y)
                nx2 = min(w, x2 + pad_x)
                ny2 = min(h, y2 + pad_y)
                
                placa_roi = frame[ny1:ny2, nx1:nx2]
                
                if placa_roi.size == 0 or placa_roi.shape[0] < 10 or placa_roi.shape[1] < 20:
                    continue
                
                # ── Redimensionar ROI para OCR óptimo ──
                roi_h, roi_w = placa_roi.shape[:2]
                if roi_w < 200:
                    scale = 200 / roi_w
                    placa_roi = cv2.resize(placa_roi, None, fx=scale, fy=scale, 
                                          interpolation=cv2.INTER_CUBIC)
                
                # ── STAGE 1: Multi-Pipeline OCR (4 estrategias) ──
                pipelines = [
                    ("CLAHE", self._pipeline_clahe),
                    ("Bilateral", self._pipeline_bilateral),
                    ("Otsu", self._pipeline_otsu),
                    ("Morph", self._pipeline_morph),
                ]
                
                mejor_texto_frame = None
                mejor_conf_ocr = 0
                
                for nombre, pipeline_fn in pipelines:
                    try:
                        roi_procesada = pipeline_fn(placa_roi)
                        resultado_ocr = self.reader.readtext(roi_procesada)
                        
                        if not resultado_ocr:
                            continue
                        
                        # Tomar el texto con mayor confianza OCR
                        for deteccion in resultado_ocr:
                            texto_crudo = deteccion[1]
                            conf_ocr = deteccion[2]
                            
                            placa_validada, categoria = self._validar_placa(texto_crudo)
                            
                            if placa_validada == "INVALID_PLATE":
                                continue
                            
                            # Bonus de confianza si matchea formato peruano exacto
                            bonus = 0.15 if categoria in ["PARTICULAR", "COMERCIAL", "MOTO", "ESPECIAL"] else 0
                            conf_total = conf_ocr + bonus
                            
                            if conf_total > mejor_conf_ocr:
                                mejor_conf_ocr = conf_total
                                mejor_texto_frame = {
                                    "plate_text": placa_validada,
                                    "category": categoria,
                                    "vehicle_type": tipo_vehiculo,
                                    "pipeline": nombre,
                                }
                    except Exception as e:
                        continue
                
                # También intentar OCR directo sobre el ROI a color (fallback)
                if mejor_texto_frame is None:
                    try:
                        resultado_ocr = self.reader.readtext(placa_roi)
                        if resultado_ocr:
                            mejor_det = max(resultado_ocr, key=lambda x: x[2])
                            placa_validada, categoria = self._validar_placa(mejor_det[1])
                            if placa_validada != "INVALID_PLATE":
                                mejor_texto_frame = {
                                    "plate_text": placa_validada,
                                    "category": categoria,
                                    "vehicle_type": tipo_vehiculo,
                                    "pipeline": "DirectColor",
                                }
                                mejor_conf_ocr = mejor_det[2]
                    except:
                        pass
                
                # Evaluar si esta detección es la mejor del frame
                if mejor_texto_frame and (conf * mejor_conf_ocr) > mejor_confianza:
                    mejor_confianza = conf * mejor_conf_ocr
                    mejor_resultado = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "plate_text": mejor_texto_frame["plate_text"],
                        "category": mejor_texto_frame["category"],
                        "vehicle_type": mejor_texto_frame["vehicle_type"],
                        "confidence": round(conf, 3),
                        "ocr_confidence": round(mejor_conf_ocr, 3),
                        "pipeline_used": mejor_texto_frame["pipeline"],
                        "coordinates": [nx1, ny1, nx2, ny2]
                    }
        
        return mejor_resultado
    
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
        
        # Si no hay cámara configurada, salir
        if self.fuente_camara is None:
            return None, None
        
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
            
        # ═══ DETECCIÓN EN DOS FASES ═══
        # Fase 1: YOLO (siempre dibuja rectángulo si detecta algo)
        # Fase 2: OCR (intenta leer el texto, pero no bloquea el rectángulo)
        
        placa_detectada = None
        
        # ── FASE 1: YOLO Detection ──
        with self.lock:
            try:
                resultados = self.modelo(
                    frame,
                    conf=self.YOLO_CONF,
                    iou=self.YOLO_IOU,
                    imgsz=self.YOLO_IMGSZ,
                    verbose=False
                )
            except Exception as e:
                print(f"[YOLO ERROR] {e}")
                return frame, None
        
        if len(resultados) == 0 or len(resultados[0].boxes) == 0:
            return frame, None
        
        num_detecciones = len(resultados[0].boxes)
        print(f"[YOLO] {num_detecciones} deteccion(es) encontrada(s)")
        
        mejor_resultado = None
        mejor_confianza = 0
        
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                
                h, w = frame.shape[:2]
                box_w = x2 - x1
                box_h = y2 - y1
                
                print(f"[YOLO] Box: ({x1},{y1})-({x2},{y2}) conf={conf:.2f} cls={cls_id} size={box_w}x{box_h}")
                
                # SIEMPRE dibujar el rectángulo de detección YOLO (amarillo)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"YOLO:{conf:.0%}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Validación geométrica (flexible)
                if box_h > 0 and box_w >= self.MIN_PLATE_PIXELS:
                    ratio = box_w / box_h
                    es_placa = self.RATIO_MOTO_MIN <= ratio <= self.RATIO_AUTO_MAX
                else:
                    es_placa = False
                    ratio = 0
                
                if not es_placa:
                    print(f"[GEOM] Rechazado: ratio={ratio:.2f} (rango válido: {self.RATIO_MOTO_MIN}-{self.RATIO_AUTO_MAX})")
                    cv2.putText(frame, f"ratio:{ratio:.1f} SKIP", (x1, y2 + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    continue
                
                tipo_vehiculo = "AUTO" if ratio >= self.RATIO_AUTO_MIN else "MOTO"
                print(f"[GEOM] OK: ratio={ratio:.2f} tipo={tipo_vehiculo}")
                
                # ── FASE 2: OCR ──
                pad_x = int(box_w * 0.08)
                pad_y = int(box_h * 0.12)
                nx1 = max(0, x1 - pad_x)
                ny1 = max(0, y1 - pad_y)
                nx2 = min(w, x2 + pad_x)
                ny2 = min(h, y2 + pad_y)
                
                placa_roi = frame[ny1:ny2, nx1:nx2]
                
                if placa_roi.size == 0 or placa_roi.shape[0] < 10 or placa_roi.shape[1] < 20:
                    print(f"[OCR] ROI demasiado pequeña: {placa_roi.shape}")
                    continue
                
                # Upscale para mejor OCR
                roi_h, roi_w = placa_roi.shape[:2]
                if roi_w < 200:
                    scale = 200 / roi_w
                    placa_roi = cv2.resize(placa_roi, None, fx=scale, fy=scale, 
                                          interpolation=cv2.INTER_CUBIC)
                
                # Probar cada pipeline de preprocesamiento
                pipelines = [
                    ("CLAHE", self._pipeline_clahe),
                    ("Bilateral", self._pipeline_bilateral),
                    ("Otsu", self._pipeline_otsu),
                    ("Morph", self._pipeline_morph),
                ]
                
                mejor_texto = None
                mejor_conf_ocr = 0
                
                for nombre, pipeline_fn in pipelines:
                    try:
                        roi_procesada = pipeline_fn(placa_roi)
                        resultado_ocr = self.reader.readtext(roi_procesada)
                        
                        if not resultado_ocr:
                            continue
                        
                        for det in resultado_ocr:
                            texto_crudo = det[1]
                            conf_ocr = det[2]
                            
                            placa_val, categoria = self._validar_placa(texto_crudo)
                            
                            if placa_val == "INVALID_PLATE":
                                continue
                            
                            bonus = 0.15 if categoria in ["PARTICULAR", "COMERCIAL", "MOTO", "ESPECIAL"] else 0
                            conf_total = conf_ocr + bonus
                            
                            if conf_total > mejor_conf_ocr:
                                mejor_conf_ocr = conf_total
                                mejor_texto = {
                                    "plate_text": placa_val,
                                    "category": categoria,
                                    "vehicle_type": tipo_vehiculo,
                                    "pipeline": nombre,
                                }
                                print(f"[OCR] Pipeline {nombre}: '{texto_crudo}' → '{placa_val}' [{categoria}] conf={conf_ocr:.2f}")
                    except Exception as e:
                        continue
                
                # Fallback: OCR directo sobre color
                if mejor_texto is None:
                    try:
                        resultado_ocr = self.reader.readtext(placa_roi)
                        if resultado_ocr:
                            for det in resultado_ocr:
                                texto_crudo = det[1]
                                conf_ocr = det[2]
                                print(f"[OCR] Fallback color: '{texto_crudo}' conf={conf_ocr:.2f}")
                                
                                placa_val, categoria = self._validar_placa(texto_crudo)
                                if placa_val != "INVALID_PLATE" and conf_ocr > mejor_conf_ocr:
                                    mejor_conf_ocr = conf_ocr
                                    mejor_texto = {
                                        "plate_text": placa_val,
                                        "category": categoria,
                                        "vehicle_type": tipo_vehiculo,
                                        "pipeline": "DirectColor",
                                    }
                    except:
                        pass
                
                # Si se leyó la placa, actualizar el overlay
                if mejor_texto and (conf * mejor_conf_ocr) > mejor_confianza:
                    mejor_confianza = conf * mejor_conf_ocr
                    placa_detectada = mejor_texto["plate_text"]
                    
                    # Sobreescribir el rectángulo amarillo con VERDE
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    label = f"{placa_detectada} [{mejor_texto['category']}]"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    info = f"YOLO:{conf:.0%} | {mejor_texto['pipeline']}"
                    cv2.putText(frame, info, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    
                    # Guardar en caché
                    resultado_completo = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "plate_text": placa_detectada,
                        "category": mejor_texto["category"],
                        "vehicle_type": mejor_texto["vehicle_type"],
                        "confidence": round(conf, 3),
                        "ocr_confidence": round(mejor_conf_ocr, 3),
                        "pipeline_used": mejor_texto["pipeline"],
                        "coordinates": [nx1, ny1, nx2, ny2]
                    }
                    self.ultima_placa = resultado_completo
                    self.tiempo_ultima_placa = datetime.now()
                    print(f"[ALPR] ✓ PLACA DETECTADA: {placa_detectada} [{mejor_texto['category']}]")
                else:
                    if mejor_texto is None:
                        print(f"[OCR] Sin lectura válida para esta detección YOLO")
            
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
        # Usar obtener_frame_con_deteccion que soporta tanto webcam como IP
        frame, placa = self.obtener_frame_con_deteccion()
        if frame is None:
            return None
            
        # Si se detectó placa en el frame, el caché ya fue actualizado
        if self.ultima_placa:
            res = self.ultima_placa
            self.ultima_placa = None
            return res
            
        return None