export type Direccion = "alza" | "baja" | "sin_cambio";

export type Driver = {
  titulo: string;
  detalle: string;
  fuente: string;
  url: string;
};

export type Reporte = {
  fecha: string;
  sesion: "con_registro" | "sin_registro";
  precio_usd_oz: number | null;
  precio_aproximado: boolean | null;
  precio_fuente: string | null;
  precio_url: string | null;
  cierre_previo_usd_oz: number | null;
  cierre_previo_aproximado: boolean | null;
  cambio_usd: number | null;
  cambio_pct: number | null;
  cambio_texto: string | null;
  usd_mxn: number | null;
  usd_mxn_tipo: "banxico_fix" | null;
  usd_mxn_fecha: string | null;
  usd_mxn_fuente: string | null;
  usd_mxn_url: string | null;
  mxn_por_gramo: number | null;
  mxn_por_gramo_aproximado: boolean | null;
  drivers: Driver[];
  lectura: string | null;
  notion_url: string | null;
  notas: string | null;
};

export type Punto = {
  fecha: string;
  precio: number | null;
  valor_justo: number | null;
  valor_ampliado: number | null;
  pronostico_20: number | null;
};

export type BloqueError = {
  n: number;
  mae: number | null;
  rmse: number | null;
  mae_pct: number | null;
  rmse_pct: number | null;
  mae_cero: number | null;
  rmse_cero: number | null;
  mae_cero_pct: number | null;
  rmse_cero_pct: number | null;
};

export type MetricasEsquema = {
  n: number;
  r2_cero: number | null;
  r2_media: number | null;
  rmse: number | null;
  rmse_cero: number | null;
  rmse_media: number | null;
  acierto: number | null;
  acierto_siempre_alza: number | null;
  acierto_signo_media: number | null;
};

export type FilaViva = {
  fecha: string;
  fecha_realizada: string | null;
  direccion: Direccion;
  direccion_realizada: Direccion;
  acierto: boolean;
  retorno_pronosticado: number;
  retorno_realizado: number;
  precio: number | null;
  nivel_objetivo: number | null;
  precio_realizado: number | null;
};

export type EnVivo = {
  n: number;
  aciertos: number;
  tasa: number | null;
  serie: { fecha: string; n: number; tasa: number }[];
  ultimos: FilaViva[];
};

export type Pronostico = {
  horizonte_dias_mercado: number;
  fecha: string;
  retorno_log: number;
  retorno_pct: number;
  direccion: Direccion;
  nivel_objetivo_usd: number | null;
  banda_baja_pct: number | null;
  banda_alta_pct: number | null;
  sigma_log: number | null;
  senal_debil: boolean;
  modelo: string;
};

export type HorizontePrecision = {
  metricas: { sin_solapar: MetricasEsquema; solapado: MetricasEsquema };
  errores: { sin_solapar: BloqueError; solapado: BloqueError };
  en_vivo: EnVivo;
  le_gana_al_cero: boolean;
  le_gana_al_promedio: boolean;
};

export type Modelo = {
  fecha: string;
  instrumento: string;
  instrumento_nota: string;
  precio_usd_oz: number;
  usd_mxn: number | null;
  usd_mxn_nota: string;
  mxn_por_gramo: number | null;
  valor_justo_usd: number | null;
  brecha_pct: number | null;
  valor_justo_muestra_usd: number | null;
  brecha_muestra_pct: number | null;
  n: number | null;
  inicio: string | null;
  fin: string | null;
  ampliado: {
    valor_justo_usd: number | null;
    brecha_pct: number | null;
    cierre_abs_pp: number | null;
    variables: string[];
  } | null;
  pronosticos: { "5": Pronostico; "20": Pronostico };
  precision: { "5": HorizontePrecision; "20": HorizontePrecision };
  historia: Punto[];
  fuentes: { serie: string; etiqueta: string | null; url: string | null; ultimo_dato: string | null }[];
  avisos: string[];
};

export type Snapshot = {
  generado_en: string | null;
  aviso: string;
  grafica_desde?: string;
  modelo: Modelo | null;
};
