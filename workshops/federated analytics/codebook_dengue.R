#Taller de análisis de datos federados en Salud
# Actividad de Conversión de la ficha epidemiológica de dengue a un diccionario común. 

# Pasos:
#   1. Importar la base de datos e identificar las variables
#   2. Asegurar el formato correcto de cada variable
#   3. Asignar etiquetas de variable (label variable)
#   4. Asignar etiquetas de valores (label values)
#   5. Explorar el codebook
#   6. Construir y exportar tabla del codebook

library(dplyr)
library(tidyr)
install.packages("labelled")
library(labelled)
library(tidyverse)
install.packages("writexl")
library(writexl)
install.packages("lubridate")
library(lubridate)

# 1. Importar la base e identificar las variables
dengue <- read.csv("dengue_sivigila.csv")
glimpse(dengue)

# 2. Asegurar el formato correcto de cada variable
# ¿Cuál es la importancia de este paso?

dengue <- dengue |>
  mutate(
    
    cod_eve   = as.numeric(cod_eve),
    sexo_     = as.character(sexo_),
    edad_     = as.numeric(edad_),
    uni_med_  = as.numeric(uni_med_),
        tip_cas_  = as.numeric(tip_cas_),
    clasfinal = as.numeric(clasfinal),
    
  )

glimpse(dengue)

# definir el formato de fecha
#¿qué estructura de fecha tiene la variable ini_sin?

dengue <- dengue %>%
 mutate(ini_sin_formateada = dmy(ini_sin_))

class(dengue$ini_sin_formateada)
glimpse(dengue$ini_sin_formateada)

# ¿que significa NA en clasfinal? ¿Que notan del formato de esta variable?

dengue <- dengue |>
  mutate(clasfinal = replace_na(clasfinal, 99))

glimpse(dengue$clasfinal)


# 3. Asignar etiquetas de variable (label variable)
# Las etiquetas describen QUÉ es cada variable. Tomar el texto del instructivo de la ficha de sivigila.

dengue <- dengue |>
  set_variable_labels(
    cod_eve            = "Código del evento (ítem 1.2)",
    sexo_              = "Sexo biológico del paciente (ítem 2.9)",
    edad_              = "Edad cumplida del paciente (ítem 2.6)",
    uni_med_           = "Unidad de medida de la edad (ítem 2.7)",
    ini_sin_           = "Fecha de inicio de síntomas (ítem 3.5)",
    ini_sin_formateada = "Fecha de inicio de síntomas d-m-a",
    tip_cas_           = "Clasificación inicial del caso (ítem 3.6)",
    clasfinal          = "Clasificación final del dengue (ítem 7.1)"
  )

var_label(dengue)  # etiquetas de todas las variables


# 4. Asignar etiquetas de valores (label values)
# Describen QUÉ SIGNIFICA cada código. Solo aplica a las variables categóricas.
# Sintaxis: c("Etiqueta" = código)

dengue <- dengue |>
  set_value_labels(
    cod_eve   = c("Dengue"                = 210,
                  "Dengue grave"          = 220,
                  "Mortalidad por dengue" = 580),
    uni_med_  = c("Anhos"     = 1,
                  "Meses"     = 2,
                  "Días"      = 3,
                  "Horas"     = 4,
                  "Minutos"   = 5,
                  "No aplica" = 0),
    sexo_     = c("Hombre"        = "M",
                  "Mujer"         = "F",
                  "Indeterminado" = "I"),
    tip_cas_  = c("Sospechoso"                    = 1,
                  "Probable"                      = 2,
                  "Conf.por laboratorio"          = 3,
                  "Conf. por clínica"             = 4,
                  "Conf. por nexo epidemiológico" = 5),
    clasfinal = c("No aplica"                   = 0,
                  "Dengue sin signos de alarma" = 1,
                  "Dengue con signos de alarma" = 2,
                  "Dengue grave"                = 3)
  )

val_labels(dengue)

# 5. Explorar el codebook
# Vista general de variables y etiquetas
look_for(dengue)

#   6. Construir y exportar tabla del codebook
extraer_val_labels <- function(x) {
  labs <- val_labels(x)
  if (is.null(labs) || length(labs) == 0) {
    return("Sin valores etiquetados")
  }
  paste(paste0(labs, " = ", names(labs)), collapse = "; ")
}

codebook_dengue <- tibble(
  columna             = names(dengue),
  tipo                = map_chr(dengue, ~ class(.x)[1]),
  etiqueta            = map_chr(dengue, ~ var_label(.x) %||% "Sin etiqueta"),
  valores_etiquetados = map_chr(dengue, extraer_val_labels)
)

print(codebook_dengue)

write_xlsx(codebook_dengue, "diccionario de datos_dengue.xlsx")
