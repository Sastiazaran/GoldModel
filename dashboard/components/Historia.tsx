"use client";

import { useEffect, useId, useState } from "react";
import { fechaCorta, fraccion, numero } from "../lib/format";
import type { GraficaRelato, Relato } from "../lib/types";

export function Historia({ relato }: { relato: Relato | null }) {
  const tituloId = useId();
  const [activa, setActiva] = useState(relato?.secciones[0]?.id ?? "");

  useEffect(() => {
    if (!relato) return;
    const nodos = relato.secciones
      .map((seccion) => document.getElementById(`historia-${seccion.id}`))
      .filter((nodo): nodo is HTMLElement => nodo != null);
    if (nodos.length === 0) return;
    const observador = new IntersectionObserver(
      (entradas) => {
        const visible = entradas
          .filter((entrada) => entrada.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        const id = visible?.target.id.replace("historia-", "");
        if (id) setActiva(id);
      },
      { rootMargin: "-20% 0px -55% 0px", threshold: [0.15, 0.4, 0.7] },
    );
    nodos.forEach((nodo) => observador.observe(nodo));
    return () => observador.disconnect();
  }, [relato]);

  if (!relato) {
    return (
      <section id="historia" aria-labelledby={tituloId}>
        <div className="seccion-head">
          <h2 id={tituloId}>La historia del oro</h2>
          <p>Todavía no está en el JSON. Corre el export para armarla con los números del día.</p>
        </div>
      </section>
    );
  }

  return (
    <section id="historia" aria-labelledby={tituloId} className="historia">
      <div className="seccion-head">
        <h2 id={tituloId}>{relato.titulo}</h2>
        <p>{relato.intro}</p>
      </div>
      <nav className="indice-historia" aria-label="Capítulos de la historia">
        {relato.secciones.map((seccion) => (
          <a
            key={seccion.id}
            href={`#historia-${seccion.id}`}
            aria-current={activa === seccion.id ? "true" : undefined}
          >
            <span>{seccion.paso}</span>
            {seccion.kicker}
          </a>
        ))}
      </nav>
      {relato.secciones.map((seccion) => (
        <article key={seccion.id} id={`historia-${seccion.id}`} className="capitulo">
          <p className="paso">{seccion.paso}</p>
          <p className="kicker">{seccion.kicker}</p>
          <h3>{seccion.titulo}</h3>
          {seccion.parrafos.map((parrafo, indice) => (
            <p key={`${seccion.id}-${indice}`}>{parrafo}</p>
          ))}
          {seccion.notas.length > 0 && (
            <ul className="notas-relato">
              {seccion.notas.map((nota) => (
                <li key={nota.texto}>
                  <span>{nota.texto}</span>
                  <strong>{nota.valor}</strong>
                </li>
              ))}
            </ul>
          )}
          {seccion.grafica && <MiniGrafica grafica={seccion.grafica} />}
          {seccion.enlaces.length > 0 && (
            <ul className="citas">
              {seccion.enlaces.map((enlace) => (
                <li key={enlace.texto + enlace.fecha}>
                  <p className="kicker">
                    {enlace.fecha ? fechaCorta(enlace.fecha) : "Log"} · {enlace.texto}
                  </p>
                  <p>{enlace.detalle}</p>
                  {enlace.url ? (
                    <a href={enlace.url} target="_blank" rel="noreferrer">
                      {enlace.fuente}
                      <span className="sr-only"> (se abre en otra pestaña)</span>
                    </a>
                  ) : (
                    <span>{enlace.fuente}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </article>
      ))}
    </section>
  );
}

function MiniGrafica({ grafica }: { grafica: GraficaRelato }) {
  if (grafica.tipo === "linea") return <Linea grafica={grafica} />;
  if (grafica.tipo === "barras") return <Barras grafica={grafica} />;
  return <Comparacion grafica={grafica} />;
}

function Linea({ grafica }: { grafica: Extract<GraficaRelato, { tipo: "linea" }> }) {
  const tituloId = useId();
  const puntos = grafica.puntos.filter((punto) => Number.isFinite(punto.valor));
  if (puntos.length < 2) return null;
  const ancho = 640;
  const alto = 200;
  const pad = { l: 44, r: 12, t: 16, b: 28 };
  const min = Math.min(...puntos.map((punto) => punto.valor));
  const max = Math.max(...puntos.map((punto) => punto.valor));
  const holgura = (max - min) * 0.08 || 1;
  const piso = min - holgura;
  const techo = max + holgura;
  const xDe = (indice: number) => pad.l + (indice / (puntos.length - 1)) * (ancho - pad.l - pad.r);
  const yDe = (valor: number) => pad.t + (1 - (valor - piso) / (techo - piso)) * (alto - pad.t - pad.b);
  const trazo = puntos.map((punto, indice) => `${xDe(indice).toFixed(1)},${yDe(punto.valor).toFixed(1)}`).join(" ");
  const primero = puntos[0];
  const ultimo = puntos[puntos.length - 1];
  return (
    <figure className="mini">
      <figcaption id={tituloId}>{grafica.titulo}</figcaption>
      <svg viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-labelledby={tituloId}>
        <title>
          {grafica.titulo}. De {numero(primero.valor, 0)} a {numero(ultimo.valor, 0)}.
        </title>
        <line className="guia" x1={pad.l} x2={ancho - pad.r} y1={yDe(min)} y2={yDe(min)} />
        <text className="eje" x={pad.l - 6} y={yDe(max) + 4} textAnchor="end">
          {numero(max, 0)}
        </text>
        <text className="eje" x={pad.l - 6} y={yDe(min) + 4} textAnchor="end">
          {numero(min, 0)}
        </text>
        <polyline className="linea-precio" points={trazo} />
        <circle className="marca marca-alza" cx={xDe(puntos.length - 1)} cy={yDe(ultimo.valor)} r="4" />
        <text className="eje" x={xDe(0)} y={alto - 8}>
          {fechaCorta(primero.fecha)}
        </text>
        <text className="eje" x={xDe(puntos.length - 1)} y={alto - 8} textAnchor="end">
          {fechaCorta(ultimo.fecha)}
        </text>
      </svg>
    </figure>
  );
}

function Barras({ grafica }: { grafica: Extract<GraficaRelato, { tipo: "barras" }> }) {
  const tituloId = useId();
  const puntos = grafica.puntos.filter((punto) => Number.isFinite(punto.valor));
  if (puntos.length === 0) return null;
  const tope = Math.max(...puntos.map((punto) => Math.abs(punto.valor)), 1);
  return (
    <figure className="mini">
      <figcaption id={tituloId}>{grafica.titulo}</figcaption>
      <ul className="barras" aria-labelledby={tituloId}>
        {puntos.map((punto) => (
          <li key={punto.etiqueta}>
            <span className="barra-nombre">{punto.etiqueta}</span>
            <span className="pista" aria-hidden="true">
              <span
                className={punto.valor < 0 ? "relleno baja-fondo" : "relleno"}
                style={{ width: `${(Math.abs(punto.valor) / tope) * 100}%` }}
              />
            </span>
            <span className={punto.valor < 0 ? "baja" : ""}>{numero(punto.valor, Math.abs(punto.valor) >= 100 ? 0 : 2)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

function Comparacion({ grafica }: { grafica: Extract<GraficaRelato, { tipo: "comparacion" }> }) {
  const tituloId = useId();
  const filas = grafica.filas.filter((fila) => fila.modelo != null || fila.base != null);
  if (filas.length === 0) return null;
  const tope = Math.max(
    ...filas.flatMap((fila) => [fila.modelo ?? 0, fila.base ?? 0].map((valor) => Math.abs(valor))),
    1,
  );
  return (
    <figure className="mini">
      <figcaption id={tituloId}>{grafica.titulo}</figcaption>
      <ul className="barras" aria-labelledby={tituloId}>
        {filas.map((fila) => (
          <li key={fila.etiqueta} className="par-barras">
            <span className="barra-nombre">{fila.etiqueta}</span>
            <span className="pareja">
              <span className="pista" aria-hidden="true">
                <span className="relleno" style={{ width: `${((fila.modelo ?? 0) / tope) * 100}%` }} />
              </span>
              <span>Modelo {fila.modelo == null ? "sin dato" : `${numero(fila.modelo, 2)}%`}</span>
            </span>
            <span className="pareja">
              <span className="pista" aria-hidden="true">
                <span className="relleno base" style={{ width: `${((fila.base ?? 0) / tope) * 100}%` }} />
              </span>
              <span>Sin cambio {fila.base == null ? "sin dato" : `${numero(fila.base, 2)}%`}</span>
            </span>
            <span className="acierto-mini">Acierto {fraccion(fila.acierto)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
