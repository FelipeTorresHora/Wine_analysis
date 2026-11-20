"""Gera web/public/atlas.json a partir dos CSVs de data/.

Contrato em docs/overview.md: meta, destaques, comercio, producao e preco.
No comércio exterior cada ano aparece duas vezes no cabeçalho: a primeira
coluna é quantidade (kg) e a segunda é valor (US$). Os dois ficam em campos
separados. Comercio.csv abre com separador ';' e não entra no JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

FONTE = "Embrapa Uva e Vinho"

FAMILIAS_TOTAIS = (
    "VINHO DE MESA",
    "VINHO FINO DE MESA (VINIFERA)",
    "SUCO",
)
FAMILIA_ESPUMANTE = "ESPUMANTE"

# Eslováquia, na importação, é o mesmo país que "Eslovaca, Republica" na
# exportação. Não é a Eslovênia. "Coreia, Republica Sul" é a Coreia do Sul,
# assim como as outras grafias do arquivo. Geórgia do Sul fica de fora deste
# mapa de propósito: não é a Geórgia.
CORRECOES_PAIS = {
    "Eslováquia": "Eslovaca, Republica",
    "Coreia, Republica Sul": "Coreia do Sul",
    "Coreia do Sul, Republica da": "Coreia do Sul",
    "Coreia do Sul, República": "Coreia do Sul",
}

ARQUIVOS_COMERCIO = (
    ("ImpVinhosMesa.csv", "importacao", "mesa"),
    ("ExpVinhoMesa.csv", "exportacao", "mesa"),
    ("ImpEspumantes.csv", "importacao", "espumante"),
    ("ExpEspumantes.csv", "exportacao", "espumante"),
)


def corrigir_pais(nome: str) -> str:
    nome = nome.strip()
    return CORRECOES_PAIS.get(nome, nome)


def parse_inteiro(texto: str) -> int:
    """Inteiro, aceitando o ponto de milhar de Producao.csv em 2024."""
    limpo = texto.strip().replace(".", "").replace(" ", "")
    if limpo == "":
        return 0
    if not limpo.lstrip("-").isdigit():
        raise ValueError(f"Número inválido: {texto!r}")
    return int(limpo)


def ler_comercio_exterior(
    caminho: Path, fluxo: str, produto: str
) -> tuple[list[int], list[dict]]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.reader(arquivo, delimiter="\t")
        cabecalho = next(leitor)
        if len(cabecalho) < 4 or cabecalho[0] != "Id":
            raise ValueError(f"Cabeçalho inesperado em {caminho.name}")
        resto = cabecalho[2:]
        if len(resto) % 2 != 0:
            raise ValueError(f"Anos sem par quantidade/valor em {caminho.name}")

        anos: list[int] = []
        for indice in range(0, len(resto), 2):
            quantidade, valor = resto[indice].strip(), resto[indice + 1].strip()
            if quantidade != valor or not quantidade.isdigit():
                raise ValueError(
                    f"Par de ano inválido em {caminho.name}: {quantidade!r}, {valor!r}"
                )
            anos.append(int(quantidade))
        if anos != sorted(set(anos)):
            raise ValueError(f"Anos repetidos ou fora de ordem em {caminho.name}")

        totais: dict[tuple[int, str], dict] = {}
        for linha in leitor:
            if not linha or len(linha) < 2 or not linha[1].strip():
                continue
            pais = corrigir_pais(linha[1])
            for deslocamento, ano in enumerate(anos):
                base = 2 + deslocamento * 2
                kg = parse_inteiro(linha[base] if base < len(linha) else "")
                usd = parse_inteiro(linha[base + 1] if base + 1 < len(linha) else "")
                chave = (ano, pais)
                atual = totais.get(chave)
                if atual is None:
                    totais[chave] = {
                        "ano": ano,
                        "fluxo": fluxo,
                        "produto": produto,
                        "pais": pais,
                        "kg": kg,
                        "usd": usd,
                    }
                else:
                    atual["kg"] += kg
                    atual["usd"] += usd

    registros = [item for item in totais.values() if item["kg"] or item["usd"]]
    return anos, registros


def abrir_comercio_csv(caminho: Path) -> dict:
    """Abre Comercio.csv com separador ';'. O conteúdo não vai para o JSON."""
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.reader(arquivo, delimiter=";")
        cabecalho = next(leitor)
        if "Produto" not in cabecalho or "1970" not in cabecalho:
            raise ValueError("Comercio.csv precisa do separador ';'.")
        linhas = 0
        for linha in leitor:
            if not linha:
                continue
            if len(linha) != len(cabecalho):
                raise ValueError("Linha de Comercio.csv com número de colunas diferente.")
            linhas += 1
    if linhas == 0:
        raise ValueError("Comercio.csv abriu vazio.")
    return {"colunas": cabecalho, "linhas": linhas}


def _produto_espumante(nome: str) -> bool:
    """Espumante, base de espumante e base de champanha. Frisante fica de fora."""
    texto = nome.casefold()
    return "espumante" in texto or "champanh" in texto or "champenoise" in texto


def ler_producao(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.reader(arquivo, delimiter=";")
        cabecalho = next(leitor)
        if cabecalho[:3] != ["id", "control", "produto"]:
            raise ValueError(f"Cabeçalho inesperado em {caminho.name}")
        anos = [int(coluna) for coluna in cabecalho[3:]]
        if not anos or anos != sorted(set(anos)):
            raise ValueError(f"Anos inválidos em {caminho.name}")

        totais = {familia: {} for familia in FAMILIAS_TOTAIS}
        espumante = {ano: 0 for ano in anos}
        for linha in leitor:
            if len(linha) < 3:
                continue
            controle = linha[1].strip()
            produto = linha[2].strip()
            valores = [
                parse_inteiro(linha[3 + indice] if 3 + indice < len(linha) else "")
                for indice in range(len(anos))
            ]
            if produto in totais:
                if controle != produto:
                    continue
                for ano, litros in zip(anos, valores):
                    if ano in totais[produto]:
                        raise ValueError(f"Total duplicado: {produto} {ano}")
                    totais[produto][ano] = litros
            elif _produto_espumante(produto):
                for ano, litros in zip(anos, valores):
                    espumante[ano] += litros

    for familia, serie in totais.items():
        faltam = [ano for ano in anos if ano not in serie]
        if faltam:
            raise ValueError(f"Família {familia} sem os anos {faltam[:3]}")

    registros = []
    for ano in anos:
        for familia in FAMILIAS_TOTAIS:
            registros.append(
                {"ano": ano, "familia": familia, "litros": totais[familia][ano]}
            )
        registros.append(
            {"ano": ano, "familia": FAMILIA_ESPUMANTE, "litros": espumante[ano]}
        )
    return registros


def derivar_preco(comercio: list[dict]) -> list[dict]:
    precos = []
    for registro in comercio:
        if registro["kg"] <= 0:
            continue
        precos.append(
            {
                "ano": registro["ano"],
                "fluxo": registro["fluxo"],
                "produto": registro["produto"],
                "pais": registro["pais"],
                "usd_por_kg": round(registro["usd"] / registro["kg"], 4),
            }
        )
    return precos


def _formato_percentual(razao: float) -> str:
    percentual = round(razao * 100)
    if percentual < 0:
        return f"\u2212{abs(percentual)}%"
    return f"{percentual}%"


def _formato_usd(valor: float) -> str:
    return "US$ " + f"{valor:.2f}".replace(".", ",")


def _litros(producao: list[dict], familia: str, ano: int) -> int:
    for registro in producao:
        if registro["familia"] == familia and registro["ano"] == ano:
            return registro["litros"]
    raise KeyError(f"Sem produção de {familia} em {ano}")


def montar_destaques(comercio: list[dict], producao: list[dict]) -> list[dict]:
    mesa_2024 = [
        registro
        for registro in comercio
        if registro["ano"] == 2024 and registro["produto"] == "mesa"
    ]
    importados = [registro for registro in mesa_2024 if registro["fluxo"] == "importacao"]
    exportados = [registro for registro in mesa_2024 if registro["fluxo"] == "exportacao"]
    kg_importado = sum(registro["kg"] for registro in importados)
    usd_importado = sum(registro["usd"] for registro in importados)
    kg_exportado = sum(registro["kg"] for registro in exportados)
    usd_exportado = sum(registro["usd"] for registro in exportados)
    chile = next(registro for registro in importados if registro["pais"] == "Chile")
    if kg_importado <= 0 or kg_exportado <= 0:
        raise ValueError("Comércio de vinho de mesa em 2024 sem quantidade.")

    litros_1970 = _litros(producao, "VINHO DE MESA", 1970)
    litros_2024 = _litros(producao, "VINHO DE MESA", 2024)
    preco_exportado = usd_exportado / kg_exportado
    preco_importado = usd_importado / kg_importado
    participacao = chile["kg"] / kg_importado

    return [
        {
            "id": "chile",
            "valor": _formato_percentual(participacao),
            "frase": "do vinho de mesa importado em 2024 veio do Chile.",
            "ano": 2024,
            "kg": chile["kg"],
            "kg_total": kg_importado,
            "participacao": round(participacao, 4),
        },
        {
            "id": "preco",
            "valor": _formato_usd(preco_exportado),
            "frase": (
                "o quilo que o Brasil vendeu, contra "
                f"{_formato_usd(preco_importado)} o quilo que comprou."
            ),
            "ano": 2024,
            "exportado_usd_por_kg": round(preco_exportado, 4),
            "importado_usd_por_kg": round(preco_importado, 4),
        },
        {
            "id": "mesa",
            "valor": _formato_percentual((litros_2024 - litros_1970) / litros_1970),
            "frase": (
                "a produção de vinho de mesa desde 1970, "
                "enquanto o suco de uva multiplicou."
            ),
            "litros_1970": litros_1970,
            "litros_2024": litros_2024,
            "variacao": round((litros_2024 - litros_1970) / litros_1970, 4),
        },
    ]


def construir_atlas(pasta_dados: Path) -> dict:
    abrir_comercio_csv(Path(pasta_dados) / "Comercio.csv")

    blocos: list[dict] = []
    anos_comercio: list[int] | None = None
    for nome, fluxo, produto in ARQUIVOS_COMERCIO:
        anos, registros = ler_comercio_exterior(Path(pasta_dados) / nome, fluxo, produto)
        if anos_comercio is None:
            anos_comercio = anos
        elif anos != anos_comercio:
            raise ValueError(f"Anos de {nome} diferem dos outros CSVs de comércio.")
        blocos.extend(registros)

    producao = ler_producao(Path(pasta_dados) / "Producao.csv")
    anos_producao = sorted({registro["ano"] for registro in producao})
    if anos_comercio != anos_producao:
        raise ValueError("Anos de comércio e produção não coincidem.")

    comercio = sorted(
        blocos,
        key=lambda registro: (
            registro["fluxo"],
            registro["produto"],
            registro["ano"],
            registro["pais"],
        ),
    )
    return {
        "meta": {
            "fonte": FONTE,
            "anos": anos_comercio,
            "unidades": {
                "kg": "kg",
                "usd": "US$",
                "litros": "L",
                "preco": "US$/kg",
            },
        },
        "destaques": montar_destaques(comercio, producao),
        "comercio": comercio,
        "producao": producao,
        "preco": derivar_preco(comercio),
    }


def serializar_atlas(atlas: dict) -> str:
    return json.dumps(atlas, ensure_ascii=False, indent=2) + "\n"


def gravar_atlas(atlas: dict, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(serializar_atlas(atlas), encoding="utf-8")


def main() -> None:
    raiz = Path(__file__).resolve().parents[1]
    atlas = construir_atlas(raiz / "data")
    destino = raiz / "web" / "public" / "atlas.json"
    gravar_atlas(atlas, destino)
    print(
        f"{destino}: {len(atlas['comercio'])} comércio, "
        f"{len(atlas['producao'])} produção, {len(atlas['preco'])} preços"
    )


if __name__ == "__main__":
    main()
