"""Conferências do atlas.json pedidas em docs/overview.md."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data"

_spec = importlib.util.spec_from_file_location(
    "build_atlas", RAIZ / "scripts" / "build_atlas.py"
)
build_atlas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_atlas)


def _atlas() -> dict:
    if not hasattr(_atlas, "valor"):
        _atlas.valor = build_atlas.construir_atlas(DADOS)
    return _atlas.valor


def _litros(atlas: dict, familia: str, ano: int) -> int:
    for registro in atlas["producao"]:
        if registro["familia"] == familia and registro["ano"] == ano:
            return registro["litros"]
    raise KeyError((familia, ano))


def _mesa(atlas: dict, fluxo: str, ano: int = 2024) -> list[dict]:
    return [
        registro
        for registro in atlas["comercio"]
        if registro["ano"] == ano
        and registro["fluxo"] == fluxo
        and registro["produto"] == "mesa"
    ]


class TestNumerosDeConferencia(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atlas = _atlas()

    def test_importacao_mesa_2024(self) -> None:
        registros = _mesa(self.atlas, "importacao")
        kg = sum(registro["kg"] for registro in registros)
        usd = sum(registro["usd"] for registro in registros)
        chile = next(registro for registro in registros if registro["pais"] == "Chile")
        self.assertEqual(kg, 153_122_230)
        self.assertEqual(usd, 481_082_975)
        self.assertEqual(chile["kg"], 73_111_416)
        self.assertEqual(round(chile["kg"] / kg * 100), 48)
        self.assertIsInstance(chile["kg"], int)
        self.assertIsInstance(chile["usd"], int)

    def test_exportacao_mesa_2024(self) -> None:
        registros = _mesa(self.atlas, "exportacao")
        kg = sum(registro["kg"] for registro in registros)
        usd = sum(registro["usd"] for registro in registros)
        paraguai = next(registro for registro in registros if registro["pais"] == "Paraguai")
        self.assertEqual(kg, 5_324_465)
        self.assertEqual(usd, 8_751_295)
        self.assertEqual(paraguai["kg"], 3_705_268)

    def test_preco_medio_mesa_2024_e_2025(self) -> None:
        importados_24 = _mesa(self.atlas, "importacao", 2024)
        exportados_24 = _mesa(self.atlas, "exportacao", 2024)
        self.assertEqual(
            round(sum(r["usd"] for r in importados_24) / sum(r["kg"] for r in importados_24), 2),
            3.14,
        )
        self.assertEqual(
            round(sum(r["usd"] for r in exportados_24) / sum(r["kg"] for r in exportados_24), 2),
            1.64,
        )
        importados = _mesa(self.atlas, "importacao", 2025)
        exportados = _mesa(self.atlas, "exportacao", 2025)
        preco_importado = sum(r["usd"] for r in importados) / sum(r["kg"] for r in importados)
        preco_exportado = sum(r["usd"] for r in exportados) / sum(r["kg"] for r in exportados)
        self.assertEqual(round(preco_importado, 2), 3.25)
        self.assertEqual(round(preco_exportado, 2), 1.66)
        destaque = next(item for item in self.atlas["destaques"] if item["id"] == "preco")
        self.assertEqual(destaque["valor"], "US$ 1,66")
        self.assertIn("US$ 3,25", destaque["frase"])
        self.assertAlmostEqual(destaque["importado_usd_por_kg"], preco_importado, places=4)
        self.assertAlmostEqual(destaque["exportado_usd_por_kg"], preco_exportado, places=4)

    def test_importacao_e_exportacao_mesa_2025(self) -> None:
        importados = _mesa(self.atlas, "importacao", 2025)
        self.assertEqual(sum(r["kg"] for r in importados), 158_532_688)
        self.assertEqual(sum(r["usd"] for r in importados), 514_800_680)
        chile = next(r for r in importados if r["pais"] == "Chile")
        self.assertEqual(chile["kg"], 77_996_609)
        self.assertEqual(round(chile["kg"] / sum(r["kg"] for r in importados) * 100), 49)

        exportados = _mesa(self.atlas, "exportacao", 2025)
        self.assertEqual(sum(r["kg"] for r in exportados), 6_540_975)
        self.assertEqual(sum(r["usd"] for r in exportados), 10_846_438)
        paraguai = next(r for r in exportados if r["pais"] == "Paraguai")
        self.assertEqual(paraguai["kg"], 4_550_344)

    def test_litros_vinho_de_mesa_1970_2024_e_2025(self) -> None:
        self.assertEqual(_litros(self.atlas, "VINHO DE MESA", 1970), 217_208_604)
        self.assertEqual(_litros(self.atlas, "VINHO DE MESA", 2024), 102_631_280)
        self.assertEqual(_litros(self.atlas, "VINHO DE MESA", 2025), 187_156_862)
        self.assertNotEqual(_litros(self.atlas, "VINHO DE MESA", 1970), 217_208_604 * 2)
        destaque = next(item for item in self.atlas["destaques"] if item["id"] == "mesa")
        self.assertEqual(destaque["litros_1970"], 217_208_604)
        self.assertEqual(destaque["litros_2025"], 187_156_862)
        self.assertEqual(destaque["valor"], "\u221214%")

    def test_ponto_de_milhar_em_2024_e_totais_2025_nas_outras_familias(self) -> None:
        self.assertEqual(_litros(self.atlas, "VINHO FINO DE MESA (VINIFERA)", 2024), 22_545_581)
        self.assertEqual(_litros(self.atlas, "SUCO", 2024), 82_884_967)
        self.assertEqual(_litros(self.atlas, "VINHO FINO DE MESA (VINIFERA)", 2025), 43_562_468)
        self.assertEqual(_litros(self.atlas, "SUCO", 2025), 89_355_889)


class TestContrato(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atlas = _atlas()

    def test_chaves_e_unidades(self) -> None:
        self.assertEqual(
            list(self.atlas),
            ["meta", "destaques", "comercio", "producao", "preco"],
        )
        self.assertEqual(set(self.atlas["meta"]), {"fonte", "anos", "unidades"})
        self.assertEqual(self.atlas["meta"]["fonte"], "Embrapa Uva e Vinho")
        self.assertEqual(self.atlas["meta"]["anos"][0], 1970)
        self.assertEqual(self.atlas["meta"]["anos"][-1], 2025)
        self.assertEqual(self.atlas["meta"]["unidades"]["preco"], "US$/kg")
        self.assertEqual(len(self.atlas["destaques"]), 3)

    def test_comercio_separa_kg_e_usd(self) -> None:
        fluxos = set()
        produtos = set()
        for registro in self.atlas["comercio"]:
            self.assertEqual(
                set(registro),
                {"ano", "fluxo", "produto", "pais", "kg", "usd"},
            )
            fluxos.add(registro["fluxo"])
            produtos.add(registro["produto"])
            self.assertIsInstance(registro["kg"], int)
            self.assertIsInstance(registro["usd"], int)
        self.assertEqual(fluxos, {"importacao", "exportacao"})
        self.assertEqual(produtos, {"mesa", "espumante"})

    def test_preco_so_quando_ha_kg(self) -> None:
        comercio = {
            (r["ano"], r["fluxo"], r["produto"], r["pais"]): r
            for r in self.atlas["comercio"]
        }
        vistos = set()
        for preco in self.atlas["preco"]:
            self.assertEqual(
                set(preco),
                {"ano", "fluxo", "produto", "pais", "usd_por_kg"},
            )
            chave = (preco["ano"], preco["fluxo"], preco["produto"], preco["pais"])
            origem = comercio[chave]
            self.assertGreater(origem["kg"], 0)
            self.assertEqual(preco["usd_por_kg"], round(origem["usd"] / origem["kg"], 4))
            vistos.add(chave)
        com_kg = {chave for chave, registro in comercio.items() if registro["kg"] > 0}
        self.assertEqual(vistos, com_kg)

    def test_producao_so_totais_de_familia(self) -> None:
        familias = {registro["familia"] for registro in self.atlas["producao"]}
        self.assertEqual(
            familias,
            {
                "VINHO DE MESA",
                "VINHO FINO DE MESA (VINIFERA)",
                "SUCO",
                "ESPUMANTE",
            },
        )
        self.assertEqual(len(self.atlas["producao"]), 56 * 4)
        texto = json.dumps(self.atlas, ensure_ascii=False)
        self.assertNotIn("vm_Tinto", texto)
        self.assertNotIn("de_Espumante", texto)
        # Soma dos espumantes em 2024, sem o pai DERIVADOS (143.197.697 L).
        # Espumante 23.994 + moscatel 10.664 + base moscatel 6.395.734
        # + Champenoise 914.741 + Charmat 3.252.935 + orgânico 140.
        self.assertEqual(_litros(self.atlas, "ESPUMANTE", 1970), 0)
        self.assertEqual(_litros(self.atlas, "ESPUMANTE", 2024), 10_598_208)
        self.assertEqual(_litros(self.atlas, "ESPUMANTE", 2025), 692_398)

    def test_comercio_csv_abre_e_nao_entra(self) -> None:
        aberto = build_atlas.abrir_comercio_csv(DADOS / "Comercio.csv")
        self.assertIn("Produto", aberto["colunas"])
        self.assertIn("1970", aberto["colunas"])
        self.assertIn("2024", aberto["colunas"])
        self.assertNotIn("2025", aberto["colunas"])
        self.assertGreater(len(aberto["colunas"]), 10)
        self.assertGreater(aberto["linhas"], 0)
        litros = {registro["litros"] for registro in self.atlas["producao"]}
        self.assertNotIn(98_327_606, litros)

    def test_arquivo_publicado_bate_com_o_script(self) -> None:
        destino = RAIZ / "web" / "public" / "atlas.json"
        self.assertEqual(
            destino.read_text(encoding="utf-8"),
            build_atlas.serializar_atlas(self.atlas),
        )


class TestPaises(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comercio = _atlas()["comercio"]
        cls.paises = {registro["pais"] for registro in cls.comercio}

    def test_eslovaquia_vira_eslovaca_republica(self) -> None:
        self.assertNotIn("Eslováquia", self.paises)
        self.assertIn("Eslovaca, Republica", self.paises)
        self.assertIn("Eslovênia", self.paises)
        self.assertTrue(
            any(
                registro["pais"] == "Eslovaca, Republica" and registro["kg"] == 1305
                for registro in self.comercio
            )
        )
        self.assertFalse(
            any(registro["pais"] == "Eslovênia" and registro["kg"] == 1305 for registro in self.comercio)
        )

    def test_georgia_do_sul_fica_separada(self) -> None:
        self.assertIn("Geórgia", self.paises)
        self.assertIn("Geórgia do Sul e Sandwich do Sul, Ilhas", self.paises)
        self.assertTrue(
            any(
                registro["pais"] == "Geórgia do Sul e Sandwich do Sul, Ilhas"
                and registro["kg"] == 1838
                for registro in self.comercio
            )
        )
        self.assertFalse(
            any(registro["pais"] == "Geórgia" and registro["kg"] == 1838 for registro in self.comercio)
        )

    def test_coreia_republica_sul_e_coreia_do_sul(self) -> None:
        self.assertNotIn("Coreia, Republica Sul", self.paises)
        self.assertNotIn("Coreia do Sul, Republica da", self.paises)
        self.assertNotIn("Coreia do Sul, República", self.paises)
        self.assertIn("Coreia do Sul", self.paises)
        self.assertTrue(
            any(
                registro["pais"] == "Coreia do Sul"
                and registro["fluxo"] == "exportacao"
                and registro["produto"] == "mesa"
                and registro["kg"] == 1012
                for registro in self.comercio
            )
        )


class TestAmostras(unittest.TestCase):
    def test_primeiro_ano_e_quantidade_e_paises(self) -> None:
        conteudo = (
            "Id\tPaís\t2024\t2024\n"
            "1\tEslováquia\t10\t30\n"
            "2\tGeórgia\t5\t15\n"
            "3\tGeórgia do Sul e Sandwich do Sul, Ilhas\t7\t21\n"
            "4\tCoreia, Republica Sul\t8\t16\n"
            "5\tChile\t0\t0\n"
            "6\tEslovaca, Republica\t1\t2\n"
            "7\tPaís sem quilo\t0\t5\n"
        )
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "amostra.csv"
            caminho.write_text(conteudo, encoding="utf-8")
            anos, registros = build_atlas.ler_comercio_exterior(
                caminho, "importacao", "mesa"
            )
        self.assertEqual(anos, [2024])
        por_pais = {registro["pais"]: registro for registro in registros}
        self.assertEqual(por_pais["Eslovaca, Republica"]["kg"], 11)
        self.assertEqual(por_pais["Eslovaca, Republica"]["usd"], 32)
        self.assertEqual(por_pais["Geórgia"]["kg"], 5)
        self.assertEqual(por_pais["Geórgia do Sul e Sandwich do Sul, Ilhas"]["kg"], 7)
        self.assertEqual(por_pais["Coreia do Sul"]["kg"], 8)
        self.assertEqual(por_pais["Coreia do Sul"]["usd"], 16)
        self.assertNotIn("Chile", por_pais)
        self.assertEqual(por_pais["País sem quilo"]["kg"], 0)
        self.assertEqual(por_pais["País sem quilo"]["usd"], 5)
        precos = {item["pais"]: item for item in build_atlas.derivar_preco(registros)}
        self.assertEqual(precos["Eslovaca, Republica"]["usd_por_kg"], round(32 / 11, 4))
        self.assertNotIn("País sem quilo", precos)
        self.assertNotIn("Chile", precos)

    def test_ponto_de_milhar_nao_soma_pai_com_filho(self) -> None:
        conteudo = "\n".join(
            [
                "id;control;produto;1970;2024",
                "1;VINHO DE MESA;VINHO DE MESA;10;102.631.280",
                "2;vm_Tinto;Tinto;7;86.404.980",
                "3;VINHO FINO DE MESA (VINIFERA);VINHO FINO DE MESA (VINIFERA);3;1.000",
                "4;SUCO;SUCO;1;2.000",
                "5;DERIVADOS;DERIVADOS;99;50.000",
                "6;de_Espumante;Espumante;0;1.500",
                "7;de_Base Charmat champanha;Base Charmat champanha;0;2.500",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "Producao.csv"
            caminho.write_text(conteudo, encoding="utf-8")
            registros = build_atlas.ler_producao(caminho)
        por_familia = {
            (registro["familia"], registro["ano"]): registro["litros"] for registro in registros
        }
        self.assertEqual(por_familia[("VINHO DE MESA", 2024)], 102_631_280)
        self.assertEqual(por_familia[("VINHO DE MESA", 1970)], 10)
        self.assertEqual(por_familia[("ESPUMANTE", 2024)], 4_000)
        self.assertEqual(por_familia[("ESPUMANTE", 1970)], 0)
        self.assertNotIn(("DERIVADOS", 2024), por_familia)
        self.assertNotIn(("Tinto", 2024), por_familia)

    def test_comercio_csv_com_virgula_falha(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "Comercio.csv"
            caminho.write_text("id,control,Produto,1970\n1,VINHO,VINHO,10\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_atlas.abrir_comercio_csv(caminho)


if __name__ == "__main__":
    unittest.main()
