from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from openai import OpenAI
import os
import fitz  # PyMuPDF
import random, string
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

#Iniciar servidor con: uvicorn main:app --reload


# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen, ajusta esto en producción
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Configura la carpeta de subida
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload_files/")
async def upload_files(files: List[UploadFile], replace: bool = Form(False)):
    saved_files = []
    updated_files = []
    duplicate_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        if os.path.exists(file_path):
            duplicate_files.append(file.filename)
        else:
            # Guardar archivo si no existe
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_files.append(file.filename)
    
    # Si hay archivos duplicados y se solicita reemplazo
    if duplicate_files and replace:
        for filename in duplicate_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            # Reemplazar archivo existente
            with open(file_path, "wb") as f:
                f.write(await [file for file in files if file.filename == filename][0].read())
            updated_files.append(filename)
        
        if saved_files:
            return JSONResponse({
                "message": f"{len(saved_files)} archivos nuevos guardados y {len(updated_files)} archivos actualizados."
            }, status_code=200)
        else:
            return JSONResponse({
                "message": f"{len(updated_files)} archivos actualizados."
            }, status_code=200)

    # Si hay archivos duplicados y NO se desea reemplazar
    if duplicate_files and not replace:
        return JSONResponse({
            "message": f"{len(saved_files)} archivos nuevos guardados, y {len(duplicate_files)} archivos duplicados no guardados.",
            "duplicates": duplicate_files
        }, status_code=200)

    # Si todos los archivos se guardaron sin duplicados
    return JSONResponse({"message": f"{len(saved_files)} archivos guardados exitosamente"}, status_code=200)

# Endpoint para eliminar todos los archivos en la carpeta uploads
@app.delete("/delete_files/")
async def delete_files():
    try:
        files_deleted = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_deleted += 1
        return JSONResponse({"message": f"{files_deleted} archivos eliminados exitosamente"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar archivos: {str(e)}")
    

# //////////////////////////////////////////
MODEL = "gpt-4o"  # Modelo especificado
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Definimos el esquema de salida de `formatear_requisitos`
class RequisitosOutput(BaseModel):
    validacion: bool
    requisitos: List[str]

# Definimos el esquema de salida de `comparar_requisitos_con_curriculum`
class ComparacionOutput(BaseModel):
    nombre: str
    dni: str
    requisitos_validados: List[int]

requisitos_procesados = []  # Variable global para almacenar los requisitos procesados
resultados = []  # Variable global para almacenar los resultados
estado_proceso = [] # Variable global para almacenar el estado del proceso

# Función para extraer texto de un archivo PDF
def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with fitz.open(ruta_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto

# Función para estructurar los requisitos usando OpenAI con salida estructurada
def formatear_requisitos(requisitos_texto: str):
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": 
                f"Analiza el siguiente texto de requisitos y estructura solo los requisitos válidos en JSON."
                f"Cada requisito debe ser concreto y resumido"
                f"Descartar aquellos requisitos que no tengan sentido o sean redundantes."
                f"Si los requisitos son válidos, devuelve 'validacion': true y lista de 'requisitos'."
                f"Si todos los requisitos no tienen sentido o no hay requisitos válidos, devuelve 'validacion': false y 'requisitos': []."
                f"Texto de requisitos: {requisitos_texto}"
            }
        ],
        response_format=RequisitosOutput,
    )

    resultado = response.choices[0].message.parsed
    return resultado  # Retorna el objeto de Pydantic ya estructurado

# Función para procesar y comparar cada currículum con salida estructurada
def comparar_requisitos_con_curriculum(requisitos_estructurados: List[str], contenido_pdf: str):
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
                {  
                "role": "user",
                "content": (
                    f"Buscaras cada requisito de la lista en el contenido del currículum"
                    f"Para cada requisito, devuelve una lista de puntuaciones donde: 0 = no cumple o no se puede determinar, 1 = cumple parcialmente, 2 = si cumple, los almacenaras en 'requisitos_validados'."
                    f"Si el requisito no especifica tiempo de experiencia o nivel de conocimiento, asume un nivel básico."
                    f"No inventes información, solo extrae la información del currículum."
                    f"Extrae el nombre completo del candidato en el contenido del currículum y lo guardarás en 'nombre', si no se encuentra pondras 'Sin nombre'."
                    f"Extrae el DNI del candidato en el contenido del currículum y lo guardarás en 'dni', si no se encuentra pondras 'Sin DNI'."
                    f"Requisitos: {requisitos_estructurados}. Contenido del currículum: {contenido_pdf}"
                )
            }
        ],
        response_format=ComparacionOutput,
    )

    resultado = response.choices[0].message.parsed
    return resultado  # Retorna el objeto de Pydantic ya estructurado

# Función principal que realiza el procesamiento
async def progreso_endpoint(requisitos: str, pdf_files: List[str]):
    global resultados, estado_proceso, requisitos_procesados
    resultados = []
    requisitos_procesados = []
    estado_proceso = ["Iniciando el proceso..."]

    # Paso 1: Formatear los requisitos
    estado_proceso.append("Procesando requisitos...")
    requisitos_procesados = formatear_requisitos(requisitos)
    
    if not requisitos_procesados.validacion:
        estado_proceso.append("Requisitos no válidos.")
        return
    else:
        estado_proceso.append("Requisitos procesados correctamente.")

    # Paso 2: Procesar cada PDF
    for pdf_file in pdf_files:
        ruta_pdf = os.path.join(UPLOAD_FOLDER, pdf_file)
        
        # Extrae el contenido del PDF
        estado_proceso.append(f"Extrayendo texto del currículum {pdf_file}...")
        contenido_pdf = extraer_texto_pdf(ruta_pdf)

        # Compara requisitos con el currículum
        estado_proceso.append(f"Comparando requisitos con el currículum {pdf_file}...")
        resultado = comparar_requisitos_con_curriculum(requisitos_procesados.requisitos, contenido_pdf)
        resultados.append(resultado.model_dump())

    estado_proceso.append("Proceso completado.")


@app.post("/procesar/")
async def procesar_requisitos(req: dict, background_tasks: BackgroundTasks):
    requisitos = req.get("requisitos","").strip()
    if not requisitos:
        raise HTTPException(status_code=400, detail="El campo de requisitos no puede estar vacío.")
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        raise HTTPException(status_code=404, detail="No hay archivos PDF en la carpeta de subida.")
    
    # Inicia el proceso en segundo plano
    background_tasks.add_task(progreso_endpoint, requisitos, pdf_files)
    return {"message": "Proceso iniciado"}

# Endpoint GET para consultar el estado del proceso
@app.get("/estado_proceso/")
async def obtener_estado_proceso():
    return {"estado_proceso": estado_proceso}

# Endpoint para mostrar los resultados

# Función para guardar datos en un archivo Excel
def guardar_datos_en_excel(resultados: List[dict]) -> str:
    carpeta = "results"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    nombre_archivo = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + ".xlsx"
    ruta_archivo = os.path.join(carpeta, nombre_archivo)

    data = []
    for resultado in resultados:
        fila = [resultado["nombre"], resultado["dni"]] + resultado["requisitos_validados"]
        data.append(fila)

    max_requisitos = max(len(r["requisitos_validados"]) for r in resultados)
    encabezados = ["nombre", "dni"] + [f"r{i+1}" for i in range(max_requisitos)]

    df = pd.DataFrame(data, columns=encabezados)
    df.to_excel(ruta_archivo, index=False)
    return ruta_archivo


@app.get("/mostrar_resultados/")
async def mostrar_resultados():
    global resultados, requisitos_procesados
    if not resultados:
        raise HTTPException(status_code=404, detail="No hay resultados disponibles.")
    
    # Ordena los resultados en base a la suma de requisitos_validados en orden descendente
    resultados = sorted(
        resultados,
        key=lambda x: sum(x["requisitos_validados"]),
        reverse=True
    )
    ruta_archivo = guardar_datos_en_excel(resultados)

    # Filtrar los resultados con las dos puntuaciones más altas después de guardar el archivo
    for res in resultados:
        res["suma_requisitos"] = sum(res["requisitos_validados"])  # Calcula la suma de requisitos_validados

    # Identifica las dos puntuaciones más altas
    puntuaciones = sorted({res["suma_requisitos"] for res in resultados}, reverse=True)
    if len(puntuaciones) > 2:
        puntuaciones = puntuaciones[:2]

    # Filtra los resultados que tengan las dos puntuaciones más altas
    resultados_filtrados = [res for res in resultados if res["suma_requisitos"] in puntuaciones]
    
    return JSONResponse({"resultados": resultados_filtrados, "requisitos_procesados": requisitos_procesados.model_dump()})