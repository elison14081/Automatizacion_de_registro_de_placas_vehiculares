from flask import Blueprint, render_template, request, jsonify, Response, current_app, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
from app import db
from app.models import Movimiento, Tarifa
from app.ocr_service import OCRService
from app.qr_service import QRService
from sqlalchemy import func
from datetime import timedelta

bp = Blueprint('main', __name__)

# ─ Normalización de placa con tolerancia OCR ───────────────────────

_OCR_EQUIV = str.maketrans('01lI', 'OOLI')   # 0→O, 1→O, l→L, I→I (normaliza)

def _normalizar(placa: str) -> str:
    """Normaliza una placa para comparación fuzzy."""
    return placa.upper().translate(_OCR_EQUIV).replace('-', '').replace(' ', '')

def _levenshtein(a: str, b: str) -> int:
    """Distancia de edición entre dos strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]

def buscar_placa_activa_similar(placa_ocr: str, umbral: int = 2):
    """
    Busca un movimiento activo cuya placa sea similar a placa_ocr
    (distancia Levenshtein ≤ umbral sobre versiones normalizadas).
    Retorna (movimiento, placa_bd) o (None, None).
    """
    norm_ocr = _normalizar(placa_ocr)
    activos = Movimiento.query.filter_by(hora_salida=None).all()
    mejor = None
    mejor_dist = umbral + 1
    for m in activos:
        dist = _levenshtein(norm_ocr, _normalizar(m.placa))
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = m
    if mejor and mejor_dist <= umbral:
        return mejor, mejor.placa
    return None, None


# Instancia global del servicio OCR
ocr_service = None

def get_ocr_service():
    global ocr_service
    if ocr_service is None:
        ocr_service = OCRService()
    return ocr_service

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/')
def index():
    """Página principal del sistema"""
    import json
    config_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        capacidad = int(cfg.get('capacidad', 60))
    except Exception:
        capacidad = 60

    try:
        ocupadas = Movimiento.query.filter_by(hora_salida=None).count()
    except Exception:
        ocupadas = 0

    disponibles = max(0, capacidad - ocupadas)
    porcentaje = int((ocupadas / capacidad) * 100) if capacidad > 0 else 0

    return render_template('index.html',
                           ocupadas=ocupadas,
                           disponibles=disponibles,
                           porcentaje=porcentaje,
                           capacidad=capacidad)

@bp.route('/estado-camara')
def estado_camara():
    """Devuelve si la cámara sigue activa en el servidor"""
    ocr = get_ocr_service()
    activa = ocr.fuente_camara is not None
    return jsonify({'activa': activa, 'fuente': str(ocr.fuente_camara) if activa else None})

# ── Páginas independientes ─────────────────────────────────────────────────────

@bp.route('/historial')
def historial():
    """Página dedicada de historial de registros"""
    return render_template('historial.html')

@bp.route('/cobros')
def cobros():
    """Página de cobros: dashboard financiero + configuración de tarifas"""
    return render_template('cobros.html')

@bp.route('/tarifas')
def tarifas():
    """Página dedicada de gestión de tarifas"""
    return render_template('tarifas.html')

@bp.route('/configuracion')
def configuracion():
    """Página dedicada de configuración del sistema"""
    import json
    config_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'config.json')
    config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
    except Exception:
        config = {}
    return render_template('configuracion.html', config=config)

@bp.route('/admin/configuracion', methods=['POST'])
def guardar_configuracion():
    """Guarda la configuración del local en un archivo JSON"""
    import json
    data = request.get_json()
    config_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'config.json')
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ELIMINAR ESTA FUNCIÓN (líneas 26-28 aproximadamente)
# @bp.route('/registro-placa')
# def registro_placa():
#     """Vista para registrar entrada de vehículos"""
#     return render_template('registro_placa.html')


@bp.route('/procesar-imagen-placa', methods=['POST'])
def procesar_imagen_placa():
    """Procesa imagen subida y detecta placa con OCR"""
    
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['imagen']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detectar placa con OCR
        ocr = get_ocr_service()
        placa = ocr.detectar_placa_desde_imagen(filepath)
        
        # Limpiar archivo temporal
        try:
            os.remove(filepath)
        except:
            pass
        
        if placa:
            return jsonify({'placa': placa, 'success': True})
        else:
            return jsonify({'error': 'No se detectó ninguna placa', 'success': False}), 404
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@bp.route('/video_feed')
def video_feed():
    """Stream de video con detección de placas en tiempo real"""
    
    # Inicializar el servicio de OCR dentro del contexto de la petición actual
    ocr = get_ocr_service()
    
    def generate():
        import time
        ocr.iniciar_camara()
        
        while True:
            try:
                frame, placa = ocr.obtener_frame_con_deteccion()
                
                if frame is None:
                    # Retornar un frame negro con mensaje de error en lugar de crashear
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Esperando senal de camara...", (100, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    time.sleep(1)
                
                # Codificar y enviar el frame (tanto válido como de error)
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"[VIDEO FEED] Error en generador: {e}")
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/capturar-placa', methods=['POST'])
def capturar_placa():
    """Captura y detecta placa desde la cámara en tiempo real"""
    
    ocr = get_ocr_service()
    res = ocr.capturar_placa()
    
    if res:
        res['success'] = True
        res['placa'] = res.get('plate_text', '')
        return jsonify(res)
    else:
        return jsonify({'error': 'No se detectó una placa', 'success': False}), 404

@bp.route('/configurar-camara', methods=['POST'])
def configurar_camara():
    """Cambia la fuente de la cámara (webcam local o IP externa)"""
    data = request.get_json() or {}
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'success': False, 'error': 'URL de cámara no proporcionada'}), 400
        
    ocr = get_ocr_service()
    exito = ocr.cambiar_camara(url)
    
    if exito:
        return jsonify({'success': True, 'message': 'Cámara conectada correctamente'})
    else:
        # Volver a cámara por defecto si falla
        ocr.cambiar_camara(0)
        return jsonify({'success': False, 'error': 'No se pudo conectar a la cámara IP'}), 400

@bp.route('/registro-placa', methods=['POST'])
def registro_placa():
    data = request.get_json() or {}
    placa = data.get('placa', request.form.get('placa', '')).strip().upper()
    
    if not placa:
        return jsonify({'success': False, 'error': 'Debe proporcionar una placa'}), 400
    
    # Buscar movimiento activo: primero coincidencia exacta, luego fuzzy
    movimiento_activo = Movimiento.query.filter_by(placa=placa, hora_salida=None).first()
    placa_bd = placa
    if not movimiento_activo:
        movimiento_activo, placa_bd = buscar_placa_activa_similar(placa)
    
    if movimiento_activo:
        # Segunda detección → calcular cobro y devolver datos de boleta
        tiempo_transcurrido = datetime.utcnow() - movimiento_activo.hora_entrada
        minutos = int(tiempo_transcurrido.total_seconds() / 60)
        monto = calcular_tarifa(minutos)
        movimiento_activo.monto = monto
        db.session.commit()
        current_app.logger.info(
            f'Checkout detectado: OCR={placa} → BD={placa_bd} '
            f'(distancia={_levenshtein(_normalizar(placa), _normalizar(placa_bd))})'
        )
        return jsonify({
            'success': False,
            'checkout': True,
            'movimiento_id': movimiento_activo.id,
            'placa': placa_bd,           # mostramos la placa original registrada
            'placa_ocr': placa,          # y la que detectó el OCR
            'hora_entrada': movimiento_activo.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
            'tiempo': f'{minutos // 60}h {minutos % 60:02d}min',
            'minutos': minutos,
            'monto': float(monto),
            'token': movimiento_activo.token
        })
    
    # Primera detección → registrar entrada
    qr_service = QRService()
    token = qr_service.generar_token()
    hora_entrada = datetime.utcnow()
    
    qr_base64 = qr_service.generar_qr_movimiento(
        placa=placa,
        token=token,
        hora_entrada=hora_entrada,
        numero_telefono=None
    )
    
    movimiento = Movimiento(
        placa=placa,
        token=token,
        hora_entrada=hora_entrada,
        qr_imagen=qr_base64,
        numero_telefono=None
    )
    
    db.session.add(movimiento)
    db.session.commit()
    
    current_app.logger.info(f'Entrada registrada: Placa {placa}, Token {token}')
    
    url_ticket = url_for('main.ver_ticket', token=token, _external=True)
    
    return jsonify({
        'success': True,
        'checkout': False,
        'placa': placa,
        'token': token,
        'qr_base64': qr_base64,
        'url_ticket': url_ticket,
        'hora_entrada': hora_entrada.strftime('%Y-%m-%d %H:%M:%S')
    })

@bp.route('/detener-camara', methods=['POST'])
def detener_camara():
    """Detiene el stream de la cámara"""
    ocr = get_ocr_service()
    ocr.detener_camara()
    return jsonify({'success': True, 'message': 'Cámara detenida'})


@bp.route('/control-salida')
def control_salida():
    """Vista para control de salida de vehículos"""
    return render_template('control_salida.html')

@bp.route('/verificar-salida-imagen', methods=['POST'])
def verificar_salida_imagen():
    """Verifica si un vehículo puede salir mediante imagen de placa"""
    
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    file = request.files['imagen']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"salida_{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detectar placa con OCR
        ocr = get_ocr_service()
        placa = ocr.detectar_placa_desde_imagen(filepath)
        
        # Limpiar archivo temporal
        try:
            os.remove(filepath)
        except:
            pass
        
        if not placa:
            return jsonify({'error': 'No se detectó ninguna placa', 'success': False}), 404
        
        # Buscar movimiento activo de este vehículo
        movimiento = Movimiento.query.filter_by(
            placa=placa,
            hora_salida=None
        ).order_by(Movimiento.hora_entrada.desc()).first()
        
        if not movimiento:
            return jsonify({
                'error': 'No hay registro de entrada para esta placa',
                'placa': placa,
                'puede_salir': False,
                'motivo': 'sin_registro'
            }), 404
        
        # Verificar si ya pagó
        resultado = {
            'placa': placa,
            'puede_salir': movimiento.pagado,
            'motivo': 'pago_ok' if movimiento.pagado else 'falta_pago',
            'movimiento': {
                'id': movimiento.id,
                'token': movimiento.token,
                'hora_entrada': movimiento.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
                'pagado': movimiento.pagado,
                'monto': float(movimiento.monto) if movimiento.monto else 0
            }
        }
        
        return jsonify(resultado)
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@bp.route('/capturar-salida', methods=['POST'])
def capturar_salida():
    """Captura placa desde cámara para verificar salida"""
    
    ocr = get_ocr_service()
    placa = ocr.capturar_placa()
    
    if not placa:
        return jsonify({'error': 'No se detectó placa', 'success': False}), 404
    
    # Buscar movimiento activo
    movimiento = Movimiento.query.filter_by(
        placa=placa,
        hora_salida=None
    ).order_by(Movimiento.hora_entrada.desc()).first()
    
    if not movimiento:
        return jsonify({
            'error': 'No hay registro de entrada para esta placa',
            'placa': placa,
            'puede_salir': False,
            'motivo': 'sin_registro'
        }), 404
    
    resultado = {
        'placa': placa,
        'puede_salir': movimiento.pagado,
        'motivo': 'pago_ok' if movimiento.pagado else 'falta_pago',
        'movimiento': {
            'id': movimiento.id,
            'token': movimiento.token,
            'hora_entrada': movimiento.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
            'pagado': movimiento.pagado,
            'monto': float(movimiento.monto) if movimiento.monto else 0
        }
    }
    
    return jsonify(resultado)

@bp.route('/autorizar-salida/<int:movimiento_id>', methods=['POST'])
def autorizar_salida(movimiento_id):
    """Autoriza la salida de un vehículo y registra hora de salida"""
    
    movimiento = Movimiento.query.get_or_404(movimiento_id)
    
    if not movimiento.pagado:
        return jsonify({'error': 'El vehículo no ha pagado'}), 400
    
    if movimiento.hora_salida:
        return jsonify({'error': 'Ya se registró la salida de este vehículo'}), 400
    
    movimiento.hora_salida = datetime.utcnow()
    
    try:
        db.session.commit()
        current_app.logger.info(f'Salida autorizada: Placa {movimiento.placa}, Token {movimiento.token}')
        
        return jsonify({
            'success': True,
            'message': 'Salida autorizada correctamente',
            'placa': movimiento.placa,
            'hora_salida': movimiento.hora_salida.strftime('%d/%m/%Y %H:%M:%S')
        })
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'Error al autorizar salida: {str(e)}')
        return jsonify({'error': 'Error al registrar salida'}), 500

@bp.route('/cobro-pago')
def cobro_pago():
    """Vista para cobro y pago de estacionamiento"""
    return render_template('cobro_pago.html')

@bp.route('/buscar-movimiento-token', methods=['POST'])
def buscar_movimiento_token():
    """Busca un movimiento por token"""
    data = request.get_json()
    token = data.get('token', '').strip()
    
    if not token:
        return jsonify({'error': 'Token no proporcionado'}), 400
    
    movimiento = Movimiento.query.filter_by(
        token=token,
        hora_salida=None
    ).first()
    
    if not movimiento:
        return jsonify({'error': 'No se encontró ningún vehículo con ese token o ya ha salido'}), 404
    
    # Calcular tiempo y costo
    tiempo_transcurrido = datetime.utcnow() - movimiento.hora_entrada
    minutos = int(tiempo_transcurrido.total_seconds() / 60)
    
    # Calcular costo según tarifas
    monto = calcular_tarifa(minutos)
    
    # Actualizar monto si no está pagado
    if not movimiento.pagado:
        movimiento.monto = monto
        db.session.commit()
    
    return jsonify({
        'id': movimiento.id,
        'placa': movimiento.placa,
        'token': movimiento.token,
        'hora_entrada': movimiento.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
        'tiempo_transcurrido': f'{minutos // 60}h {minutos % 60}min',
        'minutos': minutos,
        'monto': float(movimiento.monto) if movimiento.monto else float(monto),
        'pagado': movimiento.pagado,
        'telefono': movimiento.numero_telefono
    })

@bp.route('/buscar-movimiento-placa', methods=['POST'])
def buscar_movimiento_placa():
    """Busca un movimiento por placa"""
    data = request.get_json()
    placa = data.get('placa', '').strip().upper()
    
    if not placa:
        return jsonify({'error': 'Placa no proporcionada'}), 400
    
    movimiento = Movimiento.query.filter_by(
        placa=placa,
        hora_salida=None
    ).order_by(Movimiento.hora_entrada.desc()).first()
    
    if not movimiento:
        return jsonify({'error': 'No se encontró ningún vehículo con esa placa o ya ha salido'}), 404
    
    # Calcular tiempo y costo
    tiempo_transcurrido = datetime.utcnow() - movimiento.hora_entrada
    minutos = int(tiempo_transcurrido.total_seconds() / 60)
    
    # Calcular costo según tarifas
    monto = calcular_tarifa(minutos)
    
    # Actualizar monto si no está pagado
    if not movimiento.pagado:
        movimiento.monto = monto
        db.session.commit()
    
    return jsonify({
        'id': movimiento.id,
        'placa': movimiento.placa,
        'token': movimiento.token,
        'hora_entrada': movimiento.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
        'tiempo_transcurrido': f'{minutos // 60}h {minutos % 60}min',
        'minutos': minutos,
        'monto': float(movimiento.monto) if movimiento.monto else float(monto),
        'pagado': movimiento.pagado,
        'telefono': movimiento.numero_telefono
    })

@bp.route('/confirmar-pago', methods=['POST'])
def confirmar_pago():
    """Confirma el pago de un movimiento"""
    data = request.get_json()
    movimiento_id = data.get('movimiento_id')
    
    if not movimiento_id:
        return jsonify({'error': 'ID de movimiento no proporcionado'}), 400
    
    movimiento = Movimiento.query.get_or_404(movimiento_id)
    
    if movimiento.pagado:
        return jsonify({'error': 'Este vehículo ya ha pagado'}), 400
    
    if movimiento.hora_salida:
        return jsonify({'error': 'Este vehículo ya ha salido'}), 400
    
    # Calcular monto final
    tiempo_transcurrido = datetime.utcnow() - movimiento.hora_entrada
    minutos = int(tiempo_transcurrido.total_seconds() / 60)
    monto = calcular_tarifa(minutos)
    
    # Marcar como pagado
    movimiento.pagado = True
    movimiento.monto = monto
    
    try:
        db.session.commit()
        current_app.logger.info(f'Pago confirmado: Placa {movimiento.placa}, Monto S/ {monto}')
        
        return jsonify({
            'success': True,
            'message': 'Pago confirmado exitosamente',
            'placa': movimiento.placa,
            'monto': float(monto)
        })
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'Error al confirmar pago: {str(e)}')
        return jsonify({'error': 'Error al procesar el pago'}), 500

def calcular_tarifa(minutos):
    """Calcula la tarifa según los minutos transcurridos"""
    
    # Buscar la tarifa aplicable
    tarifa = Tarifa.query.filter(
        Tarifa.activo == True,
        Tarifa.minutos_desde <= minutos
    ).filter(
        (Tarifa.minutos_hasta >= minutos) | (Tarifa.minutos_hasta == None)
    ).order_by(Tarifa.minutos_desde.desc()).first()
    
    if tarifa:
        return tarifa.precio
    
    # Tarifa por defecto si no se encuentra ninguna
    return 5.00

# ========== RUTAS DE ADMINISTRACIÓN ==========

@bp.route('/administracion')
def administracion():
    """Vista principal de administración"""
    return render_template('administracion.html')

@bp.route('/admin/dashboard')
def admin_dashboard():
    """Datos del dashboard"""
    hoy = datetime.utcnow().date()
    
    # Estadísticas
    vehiculos_activos = Movimiento.query.filter_by(hora_salida=None).count()
    pagados_hoy = Movimiento.query.filter(
        func.date(Movimiento.hora_entrada) == hoy,
        Movimiento.pagado == True
    ).count()
    pendientes_pago = Movimiento.query.filter_by(
        hora_salida=None,
        pagado=False
    ).count()
    
    ingresos_hoy = db.session.query(func.sum(Movimiento.monto)).filter(
        func.date(Movimiento.hora_entrada) == hoy,
        Movimiento.pagado == True
    ).scalar() or 0
    
    # Vehículos recientes
    recientes = Movimiento.query.filter_by(hora_salida=None).order_by(
        Movimiento.hora_entrada.desc()
    ).limit(10).all()
    
    recientes_data = []
    for m in recientes:
        tiempo = datetime.utcnow() - m.hora_entrada
        minutos = int(tiempo.total_seconds() / 60)
        recientes_data.append({
            'placa': m.placa,
            'hora_entrada': m.hora_entrada.strftime('%d/%m/%Y %H:%M'),
            'tiempo_transcurrido': f'{minutos // 60}h {minutos % 60}min',
            'pagado': m.pagado,
            'monto': float(m.monto) if m.monto else 0
        })
    
    return jsonify({
        'vehiculos_activos': vehiculos_activos,
        'pagados_hoy': pagados_hoy,
        'pendientes_pago': pendientes_pago,
        'ingresos_hoy': float(ingresos_hoy),
        'recientes': recientes_data
    })

@bp.route('/admin/vehiculos-activos')
def admin_vehiculos_activos():
    """Lista de vehículos actualmente en el estacionamiento"""
    vehiculos = Movimiento.query.filter_by(hora_salida=None).order_by(
        Movimiento.hora_entrada.desc()
    ).all()
    
    data = []
    for v in vehiculos:
        tiempo = datetime.utcnow() - v.hora_entrada
        minutos = int(tiempo.total_seconds() / 60)
        monto = calcular_tarifa(minutos) if not v.pagado else v.monto
        
        data.append({
            'id': v.id,
            'placa': v.placa,
            'token': v.token,
            'hora_entrada': v.hora_entrada.strftime('%d/%m/%Y %H:%M:%S'),
            'tiempo_transcurrido': f'{minutos // 60}h {minutos % 60}min',
            'telefono': v.numero_telefono,
            'pagado': v.pagado,
            'monto': float(monto)
        })
    
    return jsonify(data)

@bp.route('/admin/historial')
def admin_historial():
    """Historial completo de movimientos"""
    movimientos = Movimiento.query.order_by(Movimiento.hora_entrada.desc()).limit(100).all()

    data = []
    for m in movimientos:
        duracion = None
        if m.hora_salida:
            tiempo = m.hora_salida - m.hora_entrada
            minutos = int(tiempo.total_seconds() / 60)
            duracion = f'{minutos // 60}h {minutos % 60}min'

        data.append({
            'id': m.id,
            'placa': m.placa,
            'hora_entrada': m.hora_entrada.strftime('%d/%m/%Y %H:%M'),
            'hora_salida': m.hora_salida.strftime('%d/%m/%Y %H:%M') if m.hora_salida else None,
            'duracion': duracion,
            'pagado': m.pagado,
            'monto': float(m.monto) if m.monto else 0
        })

    return jsonify(data)

@bp.route('/admin/movimiento/<int:id>', methods=['DELETE'])
def eliminar_movimiento(id):
    """Elimina un registro de movimiento por ID"""
    movimiento = Movimiento.query.get(id)
    if not movimiento:
        return jsonify({'success': False, 'error': 'Registro no encontrado'}), 404
    try:
        db.session.delete(movimiento)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/admin/movimientos/limpiar', methods=['DELETE'])
def limpiar_movimientos():
    """Elimina TODOS los registros de movimientos"""
    try:
        count = Movimiento.query.count()
        Movimiento.query.delete()
        db.session.commit()
        return jsonify({'success': True, 'eliminados': count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/admin/tarifas')
def admin_tarifas():
    """Lista de tarifas"""
    tarifas = Tarifa.query.order_by(Tarifa.minutos_desde).all()
    
    data = []
    for t in tarifas:
        data.append({
            'id': t.id,
            'descripcion': t.descripcion,
            'minutos_desde': t.minutos_desde,
            'minutos_hasta': t.minutos_hasta,
            'precio': float(t.precio),
            'activo': t.activo
        })
    
    return jsonify(data)

@bp.route('/admin/tarifa', methods=['POST'])
def admin_crear_tarifa():
    """Crear nueva tarifa"""
    data = request.get_json()
    
    tarifa = Tarifa(
        descripcion=data.get('descripcion'),
        minutos_desde=data.get('minutos_desde'),
        minutos_hasta=data.get('minutos_hasta'),
        precio=data.get('precio'),
        activo=data.get('activo', True)
    )
    
    db.session.add(tarifa)
    db.session.commit()
    
    return jsonify({'success': True, 'id': tarifa.id})

@bp.route('/admin/tarifa/<int:id>', methods=['GET'])
def admin_obtener_tarifa(id):
    """Obtener una tarifa específica"""
    tarifa = Tarifa.query.get_or_404(id)
    
    return jsonify({
        'id': tarifa.id,
        'descripcion': tarifa.descripcion,
        'minutos_desde': tarifa.minutos_desde,
        'minutos_hasta': tarifa.minutos_hasta,
        'precio': float(tarifa.precio),
        'activo': tarifa.activo
    })

@bp.route('/admin/tarifa/<int:id>', methods=['PUT'])
def admin_actualizar_tarifa(id):
    """Actualizar tarifa"""
    tarifa = Tarifa.query.get_or_404(id)
    data = request.get_json()
    
    tarifa.descripcion = data.get('descripcion')
    tarifa.minutos_desde = data.get('minutos_desde')
    tarifa.minutos_hasta = data.get('minutos_hasta')
    tarifa.precio = data.get('precio')
    tarifa.activo = data.get('activo')
    
    db.session.commit()
    
    return jsonify({'success': True})

@bp.route('/admin/tarifa/<int:id>', methods=['DELETE'])
def admin_eliminar_tarifa(id):
    """Eliminar tarifa"""
    tarifa = Tarifa.query.get_or_404(id)
    db.session.delete(tarifa)
    db.session.commit()
    
    return jsonify({'success': True})

@bp.route('/admin/reportes')
def admin_reportes():
    """Datos para reportes"""
    hoy = datetime.utcnow().date()
    hace_semana = hoy - timedelta(days=7)
    hace_mes = hoy - timedelta(days=30)
    
    ingresos_hoy = db.session.query(func.sum(Movimiento.monto)).filter(
        func.date(Movimiento.hora_entrada) == hoy,
        Movimiento.pagado == True
    ).scalar() or 0
    
    ingresos_semana = db.session.query(func.sum(Movimiento.monto)).filter(
        func.date(Movimiento.hora_entrada) >= hace_semana,
        Movimiento.pagado == True
    ).scalar() or 0
    
    ingresos_mes = db.session.query(func.sum(Movimiento.monto)).filter(
        func.date(Movimiento.hora_entrada) >= hace_mes,
        Movimiento.pagado == True
    ).scalar() or 0
    
    total_vehiculos = Movimiento.query.filter_by(pagado=True).count()
    
    return jsonify({
        'ingresos_hoy': float(ingresos_hoy),
        'ingresos_semana': float(ingresos_semana),
        'ingresos_mes': float(ingresos_mes),
        'total_vehiculos': total_vehiculos
    })
@bp.route('/mi-ticket/<token>')
def ver_ticket(token):
    """Vista pública para que el usuario vea su ticket al escanear el QR"""
    
    # Buscar el movimiento por token
    movimiento = Movimiento.query.filter_by(token=token).first()
    
    if not movimiento:
        return render_template('error.html',
            mensaje='❌ Ticket no encontrado',
            detalle='El código QR escaneado no es válido o ha expirado.')
    
    # Calcular tiempo transcurrido
    ahora = datetime.utcnow()
    
    if movimiento.hora_salida:
        # Ya salió
        tiempo_total = movimiento.hora_salida - movimiento.hora_entrada
        estado = 'finalizado'
    else:
        # Aún en el estacionamiento
        tiempo_total = ahora - movimiento.hora_entrada
        estado = 'activo'
    
    minutos_total = int(tiempo_total.total_seconds() / 60)
    horas = minutos_total // 60
    minutos = minutos_total % 60
    
    # Calcular monto actual si no ha salido
    if not movimiento.hora_salida:
        monto_actual = calcular_tarifa(minutos_total)
    else:
        monto_actual = movimiento.monto
    
    # Datos para el template
    datos_ticket = {
        'movimiento': movimiento,
        'estado': estado,
        'tiempo_texto': f'{horas}h {minutos}min',
        'monto_actual': float(monto_actual) if monto_actual else 0,
        'qr_base64': movimiento.qr_imagen,
        'fecha_consulta': ahora  # ← AGREGAR ESTA LÍNEA
    }
    
    return render_template('ver_ticket.html', **datos_ticket)