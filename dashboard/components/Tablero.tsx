"use client";

import { useMemo, useState } from "react";
import { GraficaAcierto, GraficaPrecio } from "./Graficas";
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
        <a href="#reporte">Reporte</a>
        <a href="#pronostico">Pronóstico</a>
        <a href="#precision">Precisión</a>
      </nav>

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
          <ModeloVista modelo={modelo} mostrarAmpliado={mostrarAmpliado} onAmpliado={setMostrarAmpliado} />
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
}: {
  modelo: Modelo;
  mostrarAmpliado: boolean;
  onAmpliado: (valor: boolean) => void;
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
          <p className="kicker">Valor justo, solo el pasado</p>
          <p className="grande">{modelo.valor_justo_usd == null ? "Sin dato" : usd(modelo.valor_justo_usd, 0)}</p>
          <p className="unidad">
            {modelo.brecha_pct == null ? "Sin dato" : textoBrecha(modelo.brecha_pct)}
          </p>
          {modelo.valor_justo_muestra_usd != null && modelo.brecha_muestra_pct != null && (
            <p className="detalle-cifra">
              Con toda la historia: {usd(modelo.valor_justo_muestra_usd, 0)}, brecha {puntos(modelo.brecha_muestra_pct, 1)}.
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
      <div className="grafica-head">
        <h3>Precio, valor justo y a dónde apuntaba el pronóstico</h3>
        <label>
          <input
            type="checkbox"
            checked={mostrarAmpliado}
            onChange={(evento) => onAmpliado(evento.target.checked)}
          />
          Ver el valor justo ampliado
        </label>
      </div>
      <GraficaPrecio
        puntos={modelo.historia}
        marcas={[
          { horizonte: 5, nivel: cinco.nivel_objetivo_usd, direccion: cinco.direccion },
          { horizonte: 20, nivel: veinte.nivel_objetivo_usd, direccion: veinte.direccion },
        ]}
        mostrarAmpliado={mostrarAmpliado}
        desde={modelo.historia[0]?.fecha ?? modelo.fecha}
      />
      <p className="nota">
        La línea punteada es el nivel que el pronóstico de 20 días, hecho 20 sesiones antes, apuntaba para ese día. Los puntos de la derecha son el pronóstico que todavía no vence. La gráfica empieza en {modelo.historia[0] ? fechaMedia(modelo.historia[0].fecha) : "la muestra exportada"}; las métricas usan toda la prueba fuera de muestra.
      </p>
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
      <div className="tabla-wrap">
        <table>
          <caption>
            MAE y RMSE en puntos porcentuales aproximados del retorno logarítmico (el error × 100). El acierto es la fracción de signos que coinciden.
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
