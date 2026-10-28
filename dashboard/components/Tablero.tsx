"use client";

import { useMemo, useState } from "react";
import { GraficaAcierto, GraficaPrecio } from "./Graficas";
import { Historia } from "./Historia";
import {
  cuentaGramos,
  fechaCorta,
  fechaLarga,
  fechaMedia,
  fraccion,
  instante,
  numero,
  pesos,
  puntos,
  r2,
  textoBrecha,
  textoDireccion,
  usd,
} from "../lib/format";
import type { HorizontePrecision, Modelo, Pronostico, Reporte, Snapshot } from "../lib/types";

type Props = {
  reportes: Reporte[];
  snapshot: Snapshot;
};

export function Tablero({ reportes, snapshot }: Props) {
  const inicial = useMemo(() => ultimoConRegistro(reportes), [reportes]);
  const [fecha, setFecha] = useState(inicial);
  const [mostrarAmpliado, setMostrarAmpliado] = useState(false);
  const [mostrarNuevo, setMostrarNuevo] = useState(false);
  const reporte = reportes.find((item) => item.fecha === fecha) ?? reportes.at(-1) ?? null;
  const modelo = snapshot.modelo;

  return (
    <div className="hoja">
      <header className="cabecera">
        <div>
          <p className="marca">Oro</p>
          <h1>Tablero del precio y del modelo</h1>
        </div>
        <p className="aviso">{snapshot.aviso}</p>
      </header>

      <nav className="anclas" aria-label="Secciones">
        <a href="#historia">Historia</a>
        <a href="#reporte">Reporte</a>
        <a href="#pronostico">Pronóstico</a>
        <a href="#direccion">Dirección</a>
        <a href="#precision">Precisión</a>
      </nav>

      <Historia relato={snapshot.relato ?? null} />

      <section id="reporte" aria-labelledby="titulo-reporte">
        <div className="seccion-head">
          <h2 id="titulo-reporte">Reporte del día</h2>
          <p>Lectura de la mañana. El spot es el de Kitco en el log, no el cierre del futuro.</p>
        </div>
        {reportes.length === 0 ? (
          <p className="vacio">No hay reportes en dashboard/data/reportes.json.</p>
        ) : (
          <>
            <div className="dias" role="group" aria-label="Elegir día">
              {reportes.map((item) => (
                <button
                  key={item.fecha}
                  type="button"
                  aria-pressed={item.fecha === reporte?.fecha}
                  className={item.sesion === "sin_registro" ? "dia dia-vacio" : "dia"}
                  onClick={() => setFecha(item.fecha)}
                >
                  <span>{fechaCorta(item.fecha)}</span>
                  {item.sesion === "sin_registro" && <small>sin dato</small>}
                </button>
              ))}
            </div>
            {reporte && <ReporteDia reporte={reporte} />}
          </>
        )}
      </section>

      <section id="pronostico" aria-labelledby="titulo-pronostico">
        <div className="seccion-head">
          <h2 id="titulo-pronostico">Pronóstico del modelo</h2>
          <p>Valor justo y retornos a 5 y 20 días de mercado. Ridge, solo con información ya conocida.</p>
        </div>
        {modelo == null ? (
          <div className="vacio">
            <p>Todavía no hay salida del modelo.</p>
            <p>Corre <code>python scripts/export_dashboard_data.py</code> y vuelve a construir el tablero.</p>
          </div>
        ) : (
          <ModeloVista
            modelo={modelo}
            mostrarAmpliado={mostrarAmpliado}
            onAmpliado={setMostrarAmpliado}
            mostrarNuevo={mostrarNuevo}
            onNuevo={setMostrarNuevo}
          />
        )}
      </section>

      <section id="precision" aria-labelledby="titulo-precision">
        <div className="seccion-head">
          <h2 id="titulo-precision">Precisión</h2>
          <p>Fuera de muestra, contra el baseline de que el precio no cambia. Una observación cada horizonte, para no contar días solapados como si fueran independientes.</p>
        </div>
        {modelo == null ? (
          <p className="vacio">Las métricas aparecen cuando el script exporta el modelo.</p>
        ) : (
          <Precision modelo={modelo} />
        )}
      </section>

      <footer className="pie">
        {snapshot.generado_en && <p>Datos del modelo exportados el {instante(snapshot.generado_en)}, hora del centro de México.</p>}
        <p>No es asesoría financiera. Una joyería en México cobra una prima encima del metal al mayoreo.</p>
      </footer>
    </div>
  );
}

function ultimoConRegistro(reportes: Reporte[]): string | null {
  for (let i = reportes.length - 1; i >= 0; i -= 1) {
    if (reportes[i].sesion === "con_registro") return reportes[i].fecha;
  }
  return reportes.at(-1)?.fecha ?? null;
}

function ReporteDia({ reporte }: { reporte: Reporte }) {
  const calculado =
    reporte.precio_usd_oz != null && reporte.usd_mxn != null
      ? cuentaGramos(reporte.precio_usd_oz, reporte.usd_mxn)
      : null;
  const claseCambio = reporte.cambio_usd == null ? "" : reporte.cambio_usd < 0 ? "baja" : "alza";

  return (
    <div aria-live="polite">
      <p className="cuando">{fechaLarga(reporte.fecha)}</p>
      {reporte.sesion === "sin_registro" && <p className="nota">{reporte.notas}</p>}
      <div className="cifras">
        <article>
          <p className="kicker">Spot</p>
          <p className="grande">{reporte.precio_usd_oz == null ? "Sin dato" : usd(reporte.precio_usd_oz)}</p>
          <p className="unidad">USD por onza{reporte.precio_aproximado ? " · aproximado" : ""}</p>
          {reporte.precio_fuente && reporte.precio_url && (
            <a href={reporte.precio_url} target="_blank" rel="noreferrer">
              {reporte.precio_fuente}
              <span className="sr-only"> (se abre en otra pestaña)</span>
            </a>
          )}
        </article>
        <article>
          <p className="kicker">Contra el cierre previo</p>
          <p className={`grande ${claseCambio}`}>
            {reporte.cambio_usd == null ? "Sin dato" : `${reporte.cambio_usd < 0 ? "−$" : "+$"}${numero(Math.abs(reporte.cambio_usd), 2)}`}
          </p>
          <p className={`unidad ${claseCambio}`}>{reporte.cambio_pct == null ? "Sin dato" : puntos(reporte.cambio_pct, 2)}</p>
          {reporte.cierre_previo_usd_oz != null && (
            <p className="detalle-cifra">
              Cierre previo {reporte.cierre_previo_aproximado ? "aprox. " : ""}
              {usd(reporte.cierre_previo_usd_oz)}
            </p>
          )}
        </article>
        <article>
          <p className="kicker">Pesos por gramo</p>
          <p className="grande">{reporte.mxn_por_gramo == null ? "Sin dato" : pesos(reporte.mxn_por_gramo)}</p>
          <p className="unidad">MXN/g{reporte.mxn_por_gramo_aproximado ? " · aproximado" : ""}</p>
          {reporte.usd_mxn != null && (
            <p className="detalle-cifra">
              {reporte.usd_mxn_fuente ?? "Tipo de cambio"} {numero(reporte.usd_mxn, 4)}
              {reporte.usd_mxn_fecha ? ` del ${fechaMedia(reporte.usd_mxn_fecha)}` : ""}
              {reporte.usd_mxn_url && (
                <>
                  {" · "}
                  <a href={reporte.usd_mxn_url} target="_blank" rel="noreferrer">
                    fuente
                    <span className="sr-only"> (se abre en otra pestaña)</span>
                  </a>
                </>
              )}
            </p>
          )}
        </article>
      </div>
      {reporte.cambio_texto && <p className="cambio-texto">{reporte.cambio_texto}</p>}
      {calculado != null && reporte.precio_usd_oz != null && reporte.usd_mxn != null && (
        <p className="formula">
          {numero(reporte.precio_usd_oz, 2)} × {numero(reporte.usd_mxn, 4)} ÷ 31.1034768 = {numero(calculado, 0)} pesos por gramo.
          {reporte.mxn_por_gramo != null && Math.abs(calculado - reporte.mxn_por_gramo) < 1.5
            ? " Coincide con la cifra del log, ya redondeada."
            : ""}
        </p>
      )}
      <div className="dos">
        <div>
          <h3>Qué lo mueve</h3>
          {reporte.drivers.length === 0 ? (
            <p className="nota">Sin drivers para este día.</p>
          ) : (
            <ul className="drivers">
              {reporte.drivers.map((driver) => (
                <li key={driver.url + driver.titulo}>
                  <h4>{driver.titulo}</h4>
                  <p>{driver.detalle}</p>
                  <a href={driver.url} target="_blank" rel="noreferrer">
                    {driver.fuente}
                    <span className="sr-only"> (se abre en otra pestaña)</span>
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="lectura">
          <h3>Lectura</h3>
          {reporte.lectura ? <p className="take">{reporte.lectura}</p> : <p className="nota">Sin lectura para este día.</p>}
          <p className="cobertura">Es una lectura con cobertura: dice qué vigilar. No es una orden de comprar o vender.</p>
          {reporte.notion_url && (
            <a href={reporte.notion_url} target="_blank" rel="noreferrer">
              Entrada en el log
              <span className="sr-only"> (se abre en otra pestaña)</span>
            </a>
          )}
        </div>
      </div>
      {reporte.sesion === "con_registro" && reporte.notas && <p className="nota">{reporte.notas}</p>}
    </div>
  );
}

function ModeloVista({
  modelo,
  mostrarAmpliado,
  onAmpliado,
  mostrarNuevo,
  onNuevo,
}: {
  modelo: Modelo;
  mostrarAmpliado: boolean;
  onAmpliado: (valor: boolean) => void;
  mostrarNuevo: boolean;
  onNuevo: (valor: boolean) => void;
}) {
  const cinco = modelo.pronosticos["5"];
  const veinte = modelo.pronosticos["20"];
  const ganaAlguno = modelo.precision["5"].le_gana_al_promedio || modelo.precision["20"].le_gana_al_promedio;

  return (
    <>
      <p className="cuando">
        Cierre del futuro {modelo.instrumento} al {fechaLarga(modelo.fecha)}
      </p>
      <p className="nota">{modelo.instrumento_nota}</p>
      {!ganaAlguno && (
        <p className="banda">
          Ni a 5 ni a 20 días el modelo le gana al promedio histórico del retorno. La flecha es la cuenta. La evidencia no alcanza para tratarla como señal.
        </p>
      )}
      <div className="cifras">
        <article>
          <p className="kicker">Cierre del futuro</p>
          <p className="grande">{usd(modelo.precio_usd_oz)}</p>
          <p className="unidad">USD/oz · {modelo.instrumento}</p>
          {modelo.mxn_por_gramo != null && modelo.usd_mxn != null && (
            <p className="detalle-cifra">
              Con el tipo del panel ({numero(modelo.usd_mxn, 4)}) son {numero(modelo.mxn_por_gramo, 0)} pesos por gramo. {modelo.usd_mxn_nota}
            </p>
          )}
        </article>
        <article>
          <p className="kicker">
            {modelo.valor_justo_modelo === "ols_nuevo" ? "Valor justo nuevo, solo el pasado" : "Valor justo, solo el pasado"}
          </p>
          <p className="grande">{modelo.valor_justo_usd == null ? "Sin dato" : usd(modelo.valor_justo_usd, 0)}</p>
          <p className="unidad">
            {modelo.brecha_pct == null ? "Sin dato" : textoBrecha(modelo.brecha_pct)}
          </p>
          {modelo.valor_justo_modelo === "ols_nuevo" && modelo.valor_justo_base_usd != null && (
            <p className="detalle-cifra">
              La recta base, también solo con el pasado: {usd(modelo.valor_justo_base_usd, 0)}
              {modelo.brecha_base_pct == null ? "" : `, brecha ${puntos(modelo.brecha_base_pct, 1)}`}.
            </p>
          )}
          {modelo.valor_justo_muestra_usd != null && modelo.brecha_muestra_pct != null && (
            <p className="detalle-cifra">
              Con toda la historia, la recta base: {usd(modelo.valor_justo_muestra_usd, 0)}, brecha {puntos(modelo.brecha_muestra_pct, 1)}.
            </p>
          )}
        </article>
        <article>
          <p className="kicker">Valor justo ampliado</p>
          {modelo.ampliado?.valor_justo_usd == null ? (
            <p className="grande">Sin dato</p>
          ) : (
            <>
              <p className="grande">{usd(modelo.ampliado.valor_justo_usd, 0)}</p>
              <p className="unidad">
                {modelo.ampliado.brecha_pct == null ? "Sin dato" : textoBrecha(modelo.ampliado.brecha_pct)}
              </p>
              {modelo.ampliado.cierre_abs_pp != null && (
                <p className="detalle-cifra">{textoCierre(modelo.ampliado.cierre_abs_pp)}</p>
              )}
            </>
          )}
        </article>
      </div>
      <div className="pronosticos">
        <TarjetaPronostico pronostico={cinco} />
        <TarjetaPronostico pronostico={veinte} />
      </div>
      <Direccion modelo={modelo} />
      <div className="grafica-head">
        <h3>Precio, valor justo y a dónde apuntaba el pronóstico</h3>
        <div className="toggles">
          <label>
            <input
              type="checkbox"
              checked={mostrarAmpliado}
              onChange={(evento) => onAmpliado(evento.target.checked)}
            />
            Ver el valor justo ampliado
          </label>
          <label>
            <input
              type="checkbox"
              checked={mostrarNuevo}
              onChange={(evento) => onNuevo(evento.target.checked)}
            />
            Ver el valor justo Kalman
          </label>
        </div>
      </div>
      <GraficaPrecio
        puntos={modelo.historia}
        marcas={[
          { horizonte: 5, nivel: cinco.nivel_objetivo_usd, direccion: cinco.direccion },
          { horizonte: 20, nivel: veinte.nivel_objetivo_usd, direccion: veinte.direccion },
        ]}
        mostrarAmpliado={mostrarAmpliado}
        mostrarNuevo={mostrarNuevo}
        desde={modelo.historia[0]?.fecha ?? modelo.fecha}
        nombresRegimen={modelo.estudio?.regimen?.nombres ?? []}
      />
      <p className="nota">
        La línea punteada es el nivel que el pronóstico de 20 días, hecho 20 sesiones antes, apuntaba para ese día. Los puntos de la derecha son el pronóstico que todavía no vence. La gráfica empieza en {modelo.historia[0] ? fechaMedia(modelo.historia[0].fecha) : "la muestra exportada"}; las métricas usan toda la prueba fuera de muestra.
      </p>
    </>
  );
}

function Direccion({ modelo }: { modelo: Modelo }) {
  const estudio = modelo.estudio;
  const cinco = estudio?.clasificacion?.["5"];
  const veinte = estudio?.clasificacion?.["20"];
  if (!cinco && !veinte && !estudio?.regimen?.disponible) return null;
  return (
    <div className="direccion" id="direccion">
      <h3>Probabilidad de alza, con banda plana</h3>
      <p className="nota">
        Alza, plano o baja según una banda de media desviación de la volatilidad ya conocida. La probabilidad de alza se calibra con el tramo final del entrenamiento. No es una orden.
      </p>
      <div className="medidores">
        {cinco && <Medidor titulo="5 días" arriba={cinco.prob_up} plano={cinco.prob_plano} abajo={cinco.prob_baja} modelo={cinco.modelo} />}
        {veinte && <Medidor titulo="20 días" arriba={veinte.prob_up} plano={veinte.prob_plano} abajo={veinte.prob_baja} modelo={veinte.modelo} />}
      </div>
      {estudio?.regimen?.disponible && (
        <p>
          Régimen de hoy, por el filtro del HMM: <strong>{estudio.regimen.nombre}</strong>. {estudio.regimen.nota}
        </p>
      )}
      {estudio?.valor && (
        <p className="nota">
          La recta nueva (breakeven, bancos centrales, GPR y la dummie desde el 24 de febrero de 2022)
          {estudio.valor.valor_ols == null ? " no tiene valor." : ` da ${usd(estudio.valor.valor_ols, 0)}`}
          {estudio.valor.brecha_ols_pct == null ? "" : `, brecha ${puntos(estudio.valor.brecha_ols_pct, 1)}`}.
          {" "}
          {estudio.valor.promovido
            ? "Es el valor justo titular: el RMSE fuera de muestra, una observación cada 20 sesiones, quedó por debajo del modelo base."
            : "No reemplaza al valor justo base: no le bajó ese RMSE."}
          {estudio.valor.rmse_log_ols != null && estudio.valor.rmse_log_base != null
            ? ` RMSE del log: ${numero(estudio.valor.rmse_log_ols, 3)} contra ${numero(estudio.valor.rmse_log_base, 3)} de la recta base.`
            : ""}
          {estudio.valor.valor_kalman == null
            ? ""
            : ` El Kalman, que no compite porque persigue el precio, marca ${usd(estudio.valor.valor_kalman, 0)}.`}
          {estudio.valor.valor_markov == null
            ? ""
            : ` Markov ${usd(estudio.valor.valor_markov, 0)}${estudio.valor.brecha_markov_pct == null ? "" : `, brecha ${puntos(estudio.valor.brecha_markov_pct, 1)}`}. La muestra en la que existe no alcanza para ser titular.`}
        </p>
      )}
      <Analogos modelo={modelo} />
    </div>
  );
}

function Medidor({
  titulo,
  arriba,
  plano,
  abajo,
  modelo,
}: {
  titulo: string;
  arriba: number | null;
  plano: number | null;
  abajo: number | null;
  modelo: string | null;
}) {
  const a = arriba ?? 0;
  const p = plano ?? 0;
  const b = abajo ?? 0;
  const etiqueta = `Probabilidad a ${titulo}: alza ${fraccion(arriba)}, plano ${fraccion(plano)}, baja ${fraccion(abajo)}.`;
  return (
    <article className="tarjeta">
      <p className="kicker">{titulo} de mercado</p>
      <p className="retorno">{fraccion(arriba)}</p>
      <p className="unidad">P(alza){modelo ? ` · ${modelo}` : ""}</p>
      <div className="medidor" role="img" aria-label={etiqueta}>
        <span className="med-alza" style={{ width: `${a * 100}%` }} />
        <span className="med-plano" style={{ width: `${p * 100}%` }} />
        <span className="med-baja" style={{ width: `${b * 100}%` }} />
      </div>
      <p className="detalle-cifra">Plano {fraccion(plano)} · baja {fraccion(abajo)}</p>
    </article>
  );
}

function Analogos({ modelo }: { modelo: Modelo }) {
  const analogos = modelo.estudio?.analogos;
  if (!analogos?.disponible || !analogos.periodos?.length) return null;
  return (
    <>
      <h3>Periodos parecidos</h3>
      <p className="nota">{analogos.nota}</p>
      <ul>
        {analogos.periodos.map((periodo) => (
          <li key={`${periodo.desde}-${periodo.hasta}`}>
            {fechaMedia(periodo.desde)} a {fechaMedia(periodo.hasta)}
            {periodo.retorno_20 == null ? "" : ` · los 20 días siguientes ${puntos(periodo.retorno_20 * 100, 1)}`}
          </li>
        ))}
      </ul>
    </>
  );
}

function TablaAntesDespues({ modelo }: { modelo: Modelo }) {
  const filas = modelo.estudio?.comparacion?.filas ?? [];
  if (filas.length === 0) return null;
  return (
    <>
      <div className="tabla-wrap">
        <table>
          <caption>
            Antes y después, en las mismas fechas y sin solapar el horizonte. El titular de cada horizonte es el de menor RMSE. {modelo.estudio?.comparacion?.criterio}
          </caption>
          <thead>
            <tr>
              <th scope="col">Horizonte</th>
              <th scope="col">Modelo</th>
              <th scope="col">n</th>
              <th scope="col">RMSE</th>
              <th scope="col">RMSE sin cambio</th>
              <th scope="col">R² contra cero</th>
              <th scope="col">Acierto</th>
              <th scope="col">Siempre alza</th>
              <th scope="col">Titular</th>
            </tr>
          </thead>
          <tbody>
            {filas.map((fila) => (
              <tr key={`${fila.horizonte}-${fila.modelo}`}>
                <th scope="row">{fila.horizonte} días</th>
                <td>{fila.modelo === "ridge" ? "Ridge (antes)" : "Mediana cuantílica (después)"}</td>
                <td>{fila.n}</td>
                <td>{fila.rmse_mismas_fechas == null ? "Sin dato" : `${numero(fila.rmse_mismas_fechas * 100, 2)}%`}</td>
                <td>{fila.rmse_cero == null ? "Sin dato" : `${numero(fila.rmse_cero * 100, 2)}%`}</td>
                <td>{r2(fila.r2_cero)}</td>
                <td>{fraccion(fila.acierto)}</td>
                <td>{fraccion(fila.acierto_siempre_alza)}</td>
                <td>{fila.es_titular ? "sí" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <TablaClase modelo={modelo} />
      <Cobertura modelo={modelo} />
      <Dca modelo={modelo} />
      <Diebold modelo={modelo} />
    </>
  );
}

function TablaClase({ modelo }: { modelo: Modelo }) {
  const clasificacion = modelo.estudio?.clasificacion;
  if (!clasificacion) return null;
  const filas = (["5", "20"] as const).flatMap((horizonte) => {
    const bloque = clasificacion[horizonte];
    const elegido = bloque?.metricas?.elegido;
    if (!bloque || !elegido) return [];
    return [{ horizonte, modelo: bloque.modelo, ...elegido }];
  });
  if (filas.length === 0) return null;
  return (
    <div className="tabla-wrap">
      <table>
        <caption>
          Clasificación fuera de muestra, una observación cada horizonte. La balanced accuracy de siempre alza reparte el acierto entre alza, plano y baja. El Brier de la tasa base es el punto de comparación de la probabilidad.
        </caption>
        <thead>
          <tr>
            <th scope="col">Horizonte</th>
            <th scope="col">Modelo</th>
            <th scope="col">n</th>
            <th scope="col">Balanced accuracy</th>
            <th scope="col">Siempre alza</th>
            <th scope="col">ROC-AUC</th>
            <th scope="col">Brier</th>
            <th scope="col">Brier siempre alza</th>
            <th scope="col">Brier tasa base</th>
          </tr>
        </thead>
        <tbody>
          {filas.map((fila) => (
            <tr key={fila.horizonte}>
              <th scope="row">{fila.horizonte} días</th>
              <td>{fila.modelo ?? "Sin dato"}</td>
              <td>{fila.n}</td>
              <td>{fraccion(fila.balanced_accuracy)}</td>
              <td>{fraccion(fila.balanced_accuracy_siempre_alza)}</td>
              <td>{fila.roc_auc == null ? "Sin dato" : numero(fila.roc_auc, 3)}</td>
              <td>{fila.brier == null ? "Sin dato" : numero(fila.brier, 3)}</td>
              <td>{fila.brier_siempre_alza == null ? "Sin dato" : numero(fila.brier_siempre_alza, 3)}</td>
              <td>{fila.brier_tasa_base == null ? "Sin dato" : numero(fila.brier_tasa_base, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Cobertura({ modelo }: { modelo: Modelo }) {
  const intervalos = modelo.estudio?.intervalos;
  if (!intervalos) return null;
  const frases = (["5", "20"] as const).flatMap((horizonte) => {
    const bloque = intervalos[horizonte];
    const cob = bloque?.cobertura?.sin_solapar;
    if (!bloque || cob?.cobertura == null) return [];
    const nominal = bloque.nivel == null ? "la banda" : `la banda del ${numero(bloque.nivel * 100, 0)}%`;
    const garch = bloque.garch?.cobertura?.sin_solapar;
    const extra = garch?.cobertura == null ? "" : ` GARCH cubrió ${fraccion(garch.cobertura)} (n=${garch.n}).`;
    return [
      `A ${horizonte} días ${nominal} cubrió ${fraccion(cob.cobertura)} de los retornos ya realizados (n=${cob.n}).${extra}`,
    ];
  });
  if (frases.length === 0) return null;
  return (
    <>
      {frases.map((frase) => (
        <p key={frase} className="nota">{frase}</p>
      ))}
    </>
  );
}

function Dca({ modelo }: { modelo: Modelo }) {
  const dca = modelo.estudio?.dca;
  if (!dca?.disponible || dca.ventaja_onzas_pct == null) return null;
  return (
    <p className="nota">
      Compra en partes contra un DCA plano, con la misma caja y la P(alza) a 20 días: {puntos(dca.ventaja_onzas_pct, 2)} onzas
      ({numero(dca.onzas_partes ?? 0, 2)} contra {numero(dca.onzas_dca ?? 0, 2)}). Adelantó {dca.adelantos ?? 0} tramos y aplazó {dca.aplazos ?? 0}. {dca.nota}
    </p>
  );
}

function Diebold({ modelo }: { modelo: Modelo }) {
  const dm = modelo.estudio?.comparacion?.diebold_mariano;
  if (!dm) return null;
  const frases = (["5", "20"] as const).flatMap((horizonte) => {
    const bloque = dm[horizonte]?.nuevo_vs_ridge?.sin_solapar;
    const contraCero = dm[horizonte]?.nuevo_vs_cero?.sin_solapar;
    if (!bloque && !contraCero) return [];
    const partes = [`A ${horizonte} días, Diebold-Mariano sin solapar.`];
    if (contraCero?.p != null) {
      partes.push(
        `Contra el precio quieto, p=${numero(contraCero.p, 3)} (${contraCero.modelo_mejor ? "el modelo se equivoca menos" : "no se equivoca menos"}).`,
      );
    }
    if (bloque?.p != null) {
      partes.push(
        `Contra el Ridge, p=${numero(bloque.p, 3)} (${bloque.modelo_mejor ? "la mediana se equivoca menos" : "la mediana no se equivoca menos"}).`,
      );
    }
    return [partes.join(" ")];
  });
  if (frases.length === 0) return null;
  return (
    <>
      {frases.map((frase) => (
        <p key={frase} className="nota">{frase}</p>
      ))}
    </>
  );
}

function textoCierre(puntosCierre: number): string {
  const magnitud = numero(Math.abs(puntosCierre), 1);
  if (puntosCierre > 1) return `Esas series acercan la brecha a cero en ${magnitud} puntos.`;
  if (puntosCierre < -1) return `Esas series alejan la brecha de cero en ${magnitud} puntos.`;
  return "Esas series casi no mueven la brecha.";
}

function TarjetaPronostico({ pronostico }: { pronostico: Pronostico }) {
  const clase = pronostico.direccion === "sin_cambio" ? "" : pronostico.direccion;
  return (
    <article className={`tarjeta ${clase}`}>
      <p className="kicker">{pronostico.horizonte_dias_mercado} días de mercado</p>
      <h3 className={clase}>{textoDireccion(pronostico.direccion)}</h3>
      <p className={`retorno ${clase}`}>{puntos(pronostico.retorno_pct, 2)}</p>
      <p>
        Nivel objetivo{" "}
        <strong>{pronostico.nivel_objetivo_usd == null ? "Sin dato" : usd(pronostico.nivel_objetivo_usd)}</strong>
      </p>
      <p className="detalle-cifra">
        Banda de un error típico: {puntos(pronostico.banda_baja_pct, 1)} a {puntos(pronostico.banda_alta_pct, 1)}
      </p>
      {pronostico.senal_debil && <p className="cobertura">La señal cabe en el error típico.</p>}
      {pronostico.banda_baja_pct != null &&
        pronostico.banda_alta_pct != null &&
        pronostico.banda_baja_pct < 0 &&
        pronostico.banda_alta_pct > 0 && (
          <p className="cobertura">La banda de un error típico incluye subida y bajada.</p>
        )}
    </article>
  );
}

function Precision({ modelo }: { modelo: Modelo }) {
  return (
    <>
      <TablaAntesDespues modelo={modelo} />
      <div className="tabla-wrap">
        <table>
          <caption>
            Métricas del pronóstico titular. MAE y RMSE en puntos porcentuales aproximados del retorno logarítmico (el error × 100). El acierto es la fracción de signos que coinciden.
          </caption>
          <thead>
            <tr>
              <th scope="col">Horizonte</th>
              <th scope="col">n</th>
              <th scope="col">MAE modelo</th>
              <th scope="col">MAE sin cambio</th>
              <th scope="col">RMSE modelo</th>
              <th scope="col">RMSE sin cambio</th>
              <th scope="col">Acierto</th>
              <th scope="col">Siempre alza</th>
              <th scope="col">R² contra cero</th>
              <th scope="col">R² contra el promedio</th>
            </tr>
          </thead>
          <tbody>
            <FilaPrecision horizonte="5 días" bloque={modelo.precision["5"]} />
            <FilaPrecision horizonte="20 días" bloque={modelo.precision["20"]} />
          </tbody>
        </table>
      </div>
      <ComparacionBaseline modelo={modelo} />
      <div className="vivos">
        <BloqueVivo titulo="Acierto en vivo a 5 días" bloque={modelo.precision["5"]} />
        <BloqueVivo titulo="Acierto en vivo a 20 días" bloque={modelo.precision["20"]} />
      </div>
      {modelo.fuentes.length > 0 && (
        <ul className="fuentes">
          {modelo.fuentes.map((fuente) => (
            <li key={fuente.serie}>
              {fuente.url ? (
                <a href={fuente.url} target="_blank" rel="noreferrer">
                  {fuente.etiqueta ?? fuente.serie}
                  <span className="sr-only"> (se abre en otra pestaña)</span>
                </a>
              ) : (
                fuente.etiqueta ?? fuente.serie
              )}
              {fuente.ultimo_dato ? ` · último dato ${fechaMedia(fuente.ultimo_dato)}` : ""}
            </li>
          ))}
        </ul>
      )}
      {modelo.avisos.length > 0 && (
        <details>
          <summary>Avisos de la descarga ({modelo.avisos.length})</summary>
          <ul>
            {modelo.avisos.map((aviso) => (
              <li key={aviso}>{aviso}</li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

function ComparacionBaseline({ modelo }: { modelo: Modelo }) {
  const frases = (["5", "20"] as const).flatMap((horizonte) => {
    const error = modelo.precision[horizonte].errores.sin_solapar;
    if (error.mae_pct == null || error.mae_cero_pct == null) return [];
    if (error.mae_pct > error.mae_cero_pct) {
      return [`A ${horizonte} días el error absoluto del modelo es mayor que el de asumir que el precio no cambia.`];
    }
    if (error.mae_pct < error.mae_cero_pct) {
      return [`A ${horizonte} días el error absoluto del modelo es menor que el de asumir que el precio no cambia.`];
    }
    return [];
  });
  if (frases.length === 0) return null;
  return (
    <>
      {frases.map((frase) => (
        <p key={frase} className="nota">
          {frase}
        </p>
      ))}
    </>
  );
}

function FilaPrecision({ horizonte, bloque }: { horizonte: string; bloque: HorizontePrecision }) {
  const error = bloque.errores.sin_solapar;
  const metrica = bloque.metricas.sin_solapar;
  return (
    <tr>
      <th scope="row">{horizonte}</th>
      <td>{error.n}</td>
      <td>{error.mae_pct == null ? "Sin dato" : `${numero(error.mae_pct, 2)}%`}</td>
      <td>{error.mae_cero_pct == null ? "Sin dato" : `${numero(error.mae_cero_pct, 2)}%`}</td>
      <td>{error.rmse_pct == null ? "Sin dato" : `${numero(error.rmse_pct, 2)}%`}</td>
      <td>{error.rmse_cero_pct == null ? "Sin dato" : `${numero(error.rmse_cero_pct, 2)}%`}</td>
      <td>{fraccion(metrica.acierto)}</td>
      <td>{fraccion(metrica.acierto_siempre_alza)}</td>
      <td>{r2(metrica.r2_cero)}</td>
      <td>{r2(metrica.r2_media)}</td>
    </tr>
  );
}

function BloqueVivo({ titulo, bloque }: { titulo: string; bloque: HorizontePrecision }) {
  const vivo = bloque.en_vivo;
  const filas = [...vivo.ultimos].reverse();
  return (
    <article className="vivo">
      <h3>{titulo}</h3>
      {vivo.n === 0 ? (
        <p className="nota">Ningún pronóstico de este horizonte ha vencido todavía.</p>
      ) : (
        <>
          <p>
            {vivo.aciertos} de {vivo.n} pronósticos ya vencidos acertaron la dirección
            {vivo.tasa == null ? "" : ` (${fraccion(vivo.tasa)})`}.
          </p>
          <GraficaAcierto serie={vivo.serie} />
          <div className="tabla-wrap">
            <table>
              <caption>Los últimos pronósticos cuyo horizonte ya cerró.</caption>
              <thead>
                <tr>
                  <th scope="col">Hecho el</th>
                  <th scope="col">Venció el</th>
                  <th scope="col">Apuntaba</th>
                  <th scope="col">Pasó</th>
                  <th scope="col">Nivel</th>
                  <th scope="col">Precio al vencer</th>
                </tr>
              </thead>
              <tbody>
                {filas.map((fila) => (
                  <tr key={fila.fecha}>
                    <td>{fechaMedia(fila.fecha)}</td>
                    <td>{fila.fecha_realizada ? fechaMedia(fila.fecha_realizada) : "Sin dato"}</td>
                    <td>{fila.direccion}</td>
                    <td className={fila.acierto ? "alza" : "baja"}>{fila.acierto ? "acertó" : "falló"}</td>
                    <td>{fila.nivel_objetivo == null ? "Sin dato" : usd(fila.nivel_objetivo, 0)}</td>
                    <td>{fila.precio_realizado == null ? "Sin dato" : usd(fila.precio_realizado, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </article>
  );
}
