import requests
import base64
from flask import current_app
from datetime import datetime

class WhatsAppService:
    
    @staticmethod
    def enviar_qr_entrada(numero, placa, qr_base64, token):
        """Envía el QR de entrada al WhatsApp del cliente usando Meta API"""
        
        wa_token = current_app.config.get('WHATSAPP_TOKEN')
        wa_phone_id = current_app.config.get('WHATSAPP_PHONE_ID')
        
        if not wa_token or not wa_phone_id:
            current_app.logger.warning(f"WhatsApp API no configurada. QR no enviado.")
            return {'success': False, 'message': 'WhatsApp API no configurada', 'dev_mode': True}
        
        return WhatsAppService._enviar_con_whatsapp_api(numero, placa, qr_base64, token)
    
    @staticmethod
    def _enviar_con_whatsapp_api(numero, placa, qr_base64, token):
        """Envía mensaje con WhatsApp Business API de Meta"""
        
        wa_token = current_app.config.get('WHATSAPP_TOKEN')
        wa_phone_id = current_app.config.get('WHATSAPP_PHONE_ID')
        
        # Formatear número (asegurar que tenga código de país)
        numero_limpio = ''.join(filter(str.isdigit, numero))
        if not numero_limpio.startswith('51'):
            numero_limpio = '51' + numero_limpio
        
        current_app.logger.info(f"Intentando enviar a: {numero_limpio}")
        
        try:
            # PASO 1: Subir la imagen del QR a WhatsApp
            upload_url = f"https://graph.facebook.com/v22.0/{wa_phone_id}/media"
            
            # Convertir base64 a bytes
            qr_bytes = base64.b64decode(qr_base64)
            
            # Preparar archivo para upload
            files = {
                'file': ('qr_code.png', qr_bytes, 'image/png'),
            }
            
            headers = {
                'Authorization': f'Bearer {wa_token}'
            }
            
            data = {
                'messaging_product': 'whatsapp',
                'type': 'image/png'
            }
            
            current_app.logger.info(f"Subiendo imagen a WhatsApp Media API...")
            upload_response = requests.post(upload_url, headers=headers, data=data, files=files, timeout=30)
            
            current_app.logger.info(f"Respuesta upload: {upload_response.status_code} - {upload_response.text}")
            
            if upload_response.status_code != 200:
                error_data = upload_response.json()
                error_msg = error_data.get('error', {}).get('message', 'Error desconocido')
                
                # Verificar si es error de número no permitido
                if '131030' in str(error_data) or 'not in allowed list' in str(error_data):
                    return {
                        'success': False, 
                        'message': f'⚠️ Número {numero} no autorizado en modo desarrollo.',
                        'error_code': '131030',
                        'help': 'Debes agregar este número en Meta for Developers:\n1. Ve a https://developers.facebook.com/apps\n2. Selecciona tu app\n3. WhatsApp > API Setup\n4. En "To", agrega el número +51922394409\n5. El usuario debe enviar el código de verificación'
                    }
                
                raise Exception(f"Error al subir imagen: {upload_response.text}")
            
            # Obtener el ID de la imagen subida
            media_id = upload_response.json().get('id')
            current_app.logger.info(f"Imagen subida con ID: {media_id}")
            
            # PASO 2: Enviar mensaje con la imagen
            send_url = f"https://graph.facebook.com/v22.0/{wa_phone_id}/messages"
            
            # Mensaje de caption para la imagen
            mensaje_caption = f"""🚗 ESTACIONAMIENTO

📋 Placa: {placa}
🔑 Token: {token}
⏰ Entrada: {datetime.now().strftime('%d/%m/%Y %H:%M')}

Guarda este QR para el pago.
¡Gracias! 🅿️"""
            
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": numero_limpio,
                "type": "image",
                "image": {
                    "id": media_id,
                    "caption": mensaje_caption
                }
            }
            
            current_app.logger.info(f"Enviando mensaje con imagen...")
            send_response = requests.post(send_url, headers=headers, json=payload, timeout=30)
            
            current_app.logger.info(f"Respuesta envío: {send_response.status_code} - {send_response.text}")
            
            if send_response.status_code == 200:
                current_app.logger.info(f"✅ QR enviado exitosamente a {numero}")
                return {
                    'success': True, 
                    'message': f'QR enviado correctamente a WhatsApp +{numero_limpio}', 
                    'service': 'whatsapp_api'
                }
            else:
                error_data = send_response.json()
                error_msg = error_data.get('error', {}).get('message', 'Error desconocido')
                error_code = error_data.get('error', {}).get('code', 'N/A')
                
                current_app.logger.error(f"❌ Error al enviar mensaje: {error_msg}")
                
                # Verificar errores comunes
                if '131030' in str(error_code) or 'not in allowed list' in str(error_msg):
                    return {
                        'success': False, 
                        'message': f'⚠️ Número no autorizado. Agrega +{numero_limpio} en Meta for Developers.',
                        'error_code': error_code,
                        'help': 'Pasos:\n1. https://developers.facebook.com/apps\n2. Tu App > WhatsApp > API Setup\n3. Agregar número en "To"\n4. Usuario debe verificar con el código'
                    }
                elif '131031' in str(error_code):
                    return {
                        'success': False,
                        'message': f'⚠️ Número no válido: +{numero_limpio}',
                        'error_code': error_code
                    }
                
                raise Exception(f"Error al enviar: {send_response.text}")
            
        except requests.exceptions.Timeout:
            current_app.logger.error(f"⏱️ Timeout al comunicarse con WhatsApp API")
            return {'success': False, 'message': 'Tiempo de espera agotado al enviar WhatsApp'}
        except Exception as e:
            current_app.logger.error(f"❌ Error general al enviar WhatsApp: {str(e)}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    @staticmethod
    def verificar_numero_permitido(numero):
        """Verifica si un número está en la lista de números permitidos (modo desarrollo)"""
        
        wa_token = current_app.config.get('WHATSAPP_TOKEN')
        wa_phone_id = current_app.config.get('WHATSAPP_PHONE_ID')
        
        if not wa_token or not wa_phone_id:
            return {'permitido': False, 'message': 'API no configurada'}
        
        # Formatear número
        numero_limpio = ''.join(filter(str.isdigit, numero))
        if not numero_limpio.startswith('51'):
            numero_limpio = '51' + numero_limpio
        
        # Nota: Meta no tiene endpoint para verificar directamente
        # Solo podemos intentar enviar y manejar el error
        return {
            'permitido': True,
            'numero_formateado': numero_limpio,
            'message': 'Verificación al enviar mensaje'
        }