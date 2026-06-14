# Proyecto Final: Estimación de Esfuerzo de Software

**Alumno:** Guillermo Velázquez Rosiles — zS23014116  
**Materia:** Estadística para la Calidad de Software (NRC: 50991)  
**Catedrático:** Jenny Betsabé Vázquez Aguirre  
**Universidad Veracruzana — Facultad de Estadística e Informática**

---

## Descripción

Proyecto de estimación de esfuerzo de software mediante regresión lineal simple sobre dos conjuntos de datos con mediciones NESMA (puntos de función). Incluye análisis exploratorio, modelado predictivo y evaluación estadística con pruebas de normalidad y pruebas no paramétricas.

---

## Estructura del repositorio

```
Proyecto-estimacion-esfuerzo/
├── analisis_esfuerzo.py          
├── 429.csv                     
├── 45.csv                        
├── Figures/
└── figuras/                      
    ├── hist_A.png
    ├── hist_B.png
    ├── box_A.png
    ├── box_B.png
    ├── scatter_A.png
    ├── scatter_B.png
    ├── residuos_A.png
    ├── residuos_B.png
    ├── hist_residuos_A.png
    ├── hist_residuos_B.png
    ├── real_vs_pred_A.png
    └── real_vs_pred_B.png
```

---

## Requisitos

- Python 3.8 o superior
- Librerías:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

---

## Cómo ejecutar

```bash
# 1. Clonar el repositorio
git clone https://github.com/GuilloIV/Proyecto-estimacion-esfuerzo.git
cd Proyecto-estimacion-esfuerzo

# 2. Instalar dependencias
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# 3. Correr el script (genera la carpeta figuras/ automáticamente)
python analisis_esfuerzo.py
```

Al terminar aparecerá la carpeta `figuras/` con las 12 imágenes listas para Overleaf.

---

## Fases del proyecto

| Fase | Contenido |
|------|-----------|
| I    | Análisis Exploratorio: estadísticos descriptivos, histogramas, boxplots, dispersión y correlación |
| II   | Regresión Lineal Simple: ecuación, R², RMSE, MAE y análisis de residuos |
| III  | Evaluación Estadística: Shapiro-Wilk, Kolmogorov-Smirnov y Mann-Whitney U |
| IV   | Análisis Crítico: sobreestimación, heterocedasticidad y complejidad vs. ganancia |
| V    | Conclusiones |
