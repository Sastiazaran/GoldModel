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
  valor_nuevo?: number | null;
  valor_base?: number | null;
  pronostico_20: number | null;
  intervalo_bajo?: number | null;
  intervalo_alto?: number | null;
  regimen?: number | null;
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
  modelo?: string;
};

export type FilaComparacion = {
  horizonte: number;
  modelo: string;
  n: number;
  rmse: number | null;
  rmse_mismas_fechas: number | null;
  rmse_cero: number | null;
  r2_cero: number | null;
  r2_media: number | null;
  acierto: number | null;
  acierto_siempre_alza: number | null;
  es_titular: boolean;
};

export type MetricasClase = {
  n: number;
  balanced_accuracy: number | null;
  balanced_accuracy_siempre_alza: number | null;
  roc_auc: number | null;
  brier: number | null;
  brier_siempre_alza: number | null;
  brier_tasa_base: number | null;
  calibracion?: { n: number; p_media: number; y_media: number }[];
};

export type Estudio = {
  esquema?: string;
  avisos?: string[];
  valor?: {
    promovido: boolean;
    titular: string;
    valor_ols?: number | null;
    brecha_ols_pct?: number | null;
    valor_kalman: number | null;
    brecha_kalman_pct: number | null;
    valor_markov: number | null;
    brecha_markov_pct: number | null;
    rmse_log_base: number | null;
    rmse_log_ols?: number | null;
    rmse_log_kalman: number | null;
    rmse_log_markov: number | null;
    n_rmse: number;
    variables?: string[];
    markov_convergio?: boolean | null;
    nota?: string;
  } | null;
  comparacion?: {
    criterio: string;
    filas: FilaComparacion[];
    promovido?: Record<string, boolean>;
    diebold_mariano?: Record<
      string,
      Record<string, { sin_solapar?: { n: number; p: number | null; modelo_mejor: boolean | null; media_diferencial: number | null } }>
    >;
  } | null;
  clasificacion?: Record<
    string,
    {
      prob_up: number | null;
      prob_plano: number | null;
      prob_baja: number | null;
      modelo: string | null;
      metricas?: Record<string, MetricasClase>;
    }
  >;
  intervalos?: Record<
    string,
    {
      nivel: number | null;
      metodo: string | null;
      retorno_bajo: number | null;
      retorno_alto: number | null;
      nivel_bajo_usd: number | null;
      nivel_alto_usd: number | null;
      cobertura?: { sin_solapar?: { n: number; cobertura: number | null } };
      garch?: { cobertura?: { sin_solapar?: { n: number; cobertura: number | null } } } | null;
    }
  >;
  regimen?: {
    disponible: boolean;
    nombre?: string;
    id?: number;
    nombres?: string[];
    probabilidades?: (number | null)[];
    metodo?: string;
    nota?: string;
    ari_kmeans?: number | null;
    ari_gmm?: number | null;
  } | null;
  analogos?: {
    disponible: boolean;
    nota?: string;
    periodos?: {
      desde: string;
      hasta: string;
      distancia: number;
      retorno_20: number | null;
      retorno_60: number | null;
    }[];
  } | null;
  dca?: {
    disponible: boolean;
    nota?: string;
    n?: number;
    onzas_dca?: number;
    onzas_partes?: number;
    precio_promedio_dca?: number;
    precio_promedio_partes?: number;
    ventaja_onzas_pct?: number;
    adelantos?: number;
    aplazos?: number;
  } | null;
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
  valor_justo_modelo?: string;
  valor_justo_base_usd?: number | null;
  brecha_base_pct?: number | null;
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
  pronosticos_ridge?: { "5": Pronostico; "20": Pronostico };
  precision: { "5": HorizontePrecision; "20": HorizontePrecision };
  prob_up_5d?: number | null;
  prob_up_20d?: number | null;
  estudio?: Estudio | null;
  historia: Punto[];
  fuentes: { serie: string; etiqueta: string | null; url: string | null; ultimo_dato: string | null }[];
  avisos: string[];
};

export type NotaRelato = { texto: string; valor: string };

export type EnlaceRelato = {
  texto: string;
  detalle: string;
  fuente: string;
  url: string | null;
  fecha: string;
};

export type GraficaRelato =
  | {
      tipo: "linea";
      titulo: string;
      puntos: { fecha: string; valor: number }[];
      marcas?: { fecha: string; etiqueta: string }[];
    }
  | {
      tipo: "barras";
      titulo: string;
      puntos: { etiqueta: string; valor: number }[];
    }
  | {
      tipo: "comparacion";
      titulo: string;
      filas: { etiqueta: string; modelo: number | null; base: number | null; acierto: number | null }[];
    };

export type SeccionRelato = {
  id: string;
  paso: string;
  kicker: string;
  titulo: string;
  parrafos: string[];
  notas: NotaRelato[];
  enlaces: EnlaceRelato[];
  grafica: GraficaRelato | null;
};

export type Relato = {
  titulo: string;
  fecha: string | null;
  intro: string;
  secciones: SeccionRelato[];
};

export type Snapshot = {
  generado_en: string | null;
  aviso: string;
  grafica_desde?: string;
  modelo: Modelo | null;
  relato?: Relato | null;
};
