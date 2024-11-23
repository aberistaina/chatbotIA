from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
from typing import Dict, List, Tuple
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Inicializar el modelo y tokenizador (usando un modelo en español)
MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Constantes y configuración
CONTEXTO = {
    "personal": {
        "nombre": "Alejandro Beristain",
        "rol": "desarrollador fullstack"
    },
    "habilidades": {
        "frontend": ["React", "JavaScript", "HTML", "CSS"],
        "backend": ["Node.js", "MySQL", "PostgreSQL"]
    }
}

PATRONES_PREGUNTAS = {
    'tecnologias': r'tecnolog[ií]as?|herramientas?|frameworks?|lenguajes?',
    'experiencia': r'experiencia|conocimiento|sabe|maneja',
    'frontend': r'front|frontend|interfaz|UI',
    'backend': r'back|backend|servidor|bases? de datos',
    'general': r'quien|quién|qué hace|trabajo|perfil'
}

def analizar_pregunta(pregunta: str) -> Dict[str, bool]:
    """
    Analiza la pregunta para identificar las categorías relevantes.
    """
    tipos_identificados = {}
    pregunta_lower = pregunta.lower()
    
    for categoria, patron in PATRONES_PREGUNTAS.items():
        if re.search(patron, pregunta_lower):
            tipos_identificados[categoria] = True
            
    return tipos_identificados

def obtener_respuesta_base(tipos_pregunta: Dict[str, bool]) -> str:
    """
    Genera una respuesta base según el tipo de pregunta identificada.
    """
    if not tipos_pregunta:
        return "¿Podrías reformular tu pregunta? No estoy seguro de qué aspecto específico quieres conocer sobre Alejandro."
    
    if 'general' in tipos_pregunta:
        return (f"{CONTEXTO['personal']['nombre']} es {CONTEXTO['personal']['rol']} "
                "con experiencia en desarrollo web completo.")
    
    if 'tecnologias' in tipos_pregunta:
        if 'frontend' in tipos_pregunta:
            techs = ', '.join(CONTEXTO['habilidades']['frontend'])
            return f"En frontend, Alejandro trabaja principalmente con {techs}, siendo React su framework principal."
        
        if 'backend' in tipos_pregunta:
            techs = ', '.join(CONTEXTO['habilidades']['backend'])
            return f"En backend, Alejandro utiliza {techs} para desarrollar APIs y gestionar bases de datos."
        
        # Si pregunta por tecnologías en general
        front_techs = ', '.join(CONTEXTO['habilidades']['frontend'])
        back_techs = ', '.join(CONTEXTO['habilidades']['backend'])
        return f"Alejandro maneja tecnologías frontend como {front_techs}, y backend como {back_techs}."
    
    if 'experiencia' in tipos_pregunta:
        area = 'frontend' if 'frontend' in tipos_pregunta else 'backend' if 'backend' in tipos_pregunta else 'general'
        if area == 'general':
            return (f"{CONTEXTO['personal']['nombre']} tiene experiencia como desarrollador fullstack, "
                   "trabajando tanto en frontend como en backend.")
        techs = ', '.join(CONTEXTO['habilidades'][area])
        return f"Alejandro tiene experiencia trabajando con {techs} en el área de {area}."
    
    return "Alejandro es un desarrollador fullstack con experiencia en React y Node.js."

def generar_respuesta_modelo(prompt: str, respuesta_base: str) -> str:
    """
    Genera una respuesta usando el modelo de lenguaje.
    """
    prompt_completo = (
        f"Basándote en esta información: {respuesta_base}\n"
        f"Responde a esta pregunta: {prompt}\n"
        "Genera una respuesta concisa y relevante en español:"
    )
    
    inputs = tokenizer.encode(prompt_completo, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta

def post_procesar_respuesta(respuesta: str, respuesta_base: str) -> str:
    """
    Realiza el post-procesamiento de la respuesta para mejorar su calidad.
    """
    # Si la respuesta es muy corta o parece irrelevante, usar la respuesta base
    if len(respuesta.split()) < 5 or not any(char in respuesta.lower() for char in "áéíóúñ"):
        return respuesta_base
    
    # Eliminar repeticiones y limpiar la respuesta
    oraciones = list(dict.fromkeys(respuesta.split('. ')))
    respuesta_limpia = '. '.join(oracion.strip() for oracion in oraciones if oracion.strip())
    
    return respuesta_limpia

def procesar_pregunta(pregunta: str) -> str:
    """
    Procesa la pregunta y genera una respuesta completa.
    """
    # Analizar el tipo de pregunta
    tipos_pregunta = analizar_pregunta(pregunta)
    
    # Obtener respuesta base según el tipo de pregunta
    respuesta_base = obtener_respuesta_base(tipos_pregunta)
    
    # Generar respuesta usando el modelo
    respuesta_modelo = generar_respuesta_modelo(pregunta, respuesta_base)
    
    # Post-procesar la respuesta
    respuesta_final = post_procesar_respuesta(respuesta_modelo, respuesta_base)
    
    return respuesta_final

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Se requiere un mensaje en el campo "message"'
            }), 400
            
        mensaje_usuario = data['message']
        
        # Validar el mensaje
        if not isinstance(mensaje_usuario, str) or not mensaje_usuario.strip():
            return jsonify({
                'error': 'El mensaje debe ser un texto no vacío'
            }), 400
            
        # Generar respuesta
        respuesta = procesar_pregunta(mensaje_usuario)
        
        return jsonify({
            'response': respuesta,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la solicitud: {str(e)}',
            'status': 'error'
        }), 500

# Manejador de errores para rutas no encontradas
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Ruta no encontrada',
        'status': 'error'
    }), 404

if __name__ == '__main__':
    app.run(debug=True)