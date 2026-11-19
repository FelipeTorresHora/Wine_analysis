# Overview

Frontend do atlas da vitivinicultura brasileira, no molde da página Griffin da Tavus: uma leitura longa, com hero, três números, índice que acompanha a rolagem e gráficos no texto. Este arquivo é a lista do que muda no repositório e a ordem para fazer.

Decisão de estrutura: uma página com cinco seções, não três rotas de dashboard. Parceiros, produção e preço são âncoras. O laboratório de qualidade fica fora da primeira versão e aparece só como card na grade final.

## O que muda

- Entra uma aplicação web nova, em `web/`, que consome JSON já agregado. O visual segue `design.md`.
- Entra um script que transforma os CSVs de `data/` nesse JSON. Quantidade e valor ficam em colunas distintas. Totais de produção não se misturam com as subcategorias.
- O README passa a descrever o atlas e a apontar para a página, no lugar de `train_model.py`, `app.py` e os CSVs `*_ready.csv`, que não existem.
- A pasta `streamlit/` deixa de se chamar assim, para não esconder a biblioteca Streamlit. As páginas de lá continuam vazias e não entram nesta versão.
- `src/ml/modelo_qualidade_vinho.py` permanece fora do fluxo da página. Ele ainda aponta para `winequality-red.csv`.

## Fora desta versão

- Treino e tela do classificador de qualidade.
- Notebooks 02 a 05.
- PDF, PowerPoint e Plotly, embora estejam em `requirements.txt`.
- Cópia de texto, marca ou métricas da Tavus.

## Dados que a página precisa

Um único `web/public/atlas.json`, gerado por script, com:

- `meta`: fonte, anos, unidades.
- `destaques`: os três números do hero, já calculados, com frase.
- `comercio`: registros `{ ano, fluxo, produto, pais, kg, usd }`, fluxo `importacao` ou `exportacao`, produto `mesa` ou `espumante`.
- `producao`: registros `{ ano, familia, litros }` só para os totais `VINHO DE MESA`, `VINHO FINO DE MESA (VINIFERA)`, `SUCO` e a soma dos espumantes, sem somar o pai com os filhos.
- `preco`: US$/kg derivado de `usd / kg` quando `kg` é maior que zero.

Números de conferência, já lidos dos CSVs, para os testes do script:

- Importação de vinho de mesa em 2024: 153.122.230 kg, US$ 481.082.975. Chile: 73.111.416 kg, cerca de 48%.
- Exportação de vinho de mesa em 2024: 5.324.465 kg, US$ 8.751.275. Paraguai: 3.705.268 kg.
- Preço médio de 2024: importado cerca de US$ 3,14/kg, exportado cerca de US$ 1,64/kg.
- Produção de vinho de mesa: 217.208.604 L em 1970, 102.631.280 L em 2024.

## Tasks

### 1. Dados

- [ ] **T1.** Extrair kg e US$ dos CSVs de comércio. Cada ano vem duplicado no cabeçalho; o primeiro é quantidade, o segundo é valor.
- [ ] **T2.** Corrigir nomes de país: Eslováquia para `Eslovaca, Republica`; manter Geórgia do Sul separada da Geórgia; reconhecer `Coreia, Republica Sul`.
- [ ] **T3.** Ler `Comercio.csv` com separador `;`. Não entra no JSON da primeira versão, só precisa voltar a abrir.
- [ ] **T4.** Tratar o ponto de milhar em `Producao.csv` no ano de 2024.
- [ ] **T5.** Gerar `web/public/atlas.json` com o contrato acima e um teste que confere os totais de 2024 e os litros de 1970 e 2024.

### 2. Base da interface

- [ ] **T6.** Criar `web/` com Vite, React e TypeScript. Fontes Newsreader e Geist.
- [ ] **T7.** Aplicar os tokens, o tipo e os botões descritos em `design.md`.
- [ ] **T8.** Montar a navegação fixa, o hero e a faixa dos três números lendo `destaques` do JSON.
- [ ] **T9.** Montar o índice fixo das seções 01 a 05, com o item ativo acompanhando a rolagem.

### 3. Seções

- [ ] **T10.** Seção 01, barras horizontais de importação contra exportação em 2024.
- [ ] **T11.** Seção 02, ranking com o líder em verde e o waffle da fatia do Paraguai. Controle em pílula para fluxo e produto.
- [ ] **T12.** Seção 03, série de litros por família, só totais.
- [ ] **T13.** Seção 04, US$/kg ao longo dos anos e o recorte do espumante em 2024.
- [ ] **T14.** Seção 05, método, com a origem dos dados e o aviso sobre o WineQT.

### 4. Fecho da página

- [ ] **T15.** Grade final escura com quatro cards e a trama de pontos no hover. O card do laboratório não leva a lugar nenhum ainda.
- [ ] **T16.** Rodapé e a faixa de quadrados. A faixa respeita `prefers-reduced-motion`.
- [ ] **T17.** Empilhar índice, números e grade numa coluna estreita.

### 5. Projeto em volta

- [ ] **T18.** Reescrever o README para o atlas: como gerar o JSON, como subir a página, de onde vêm os dados.
- [ ] **T19.** Renomear a pasta `streamlit/` para `app_streamlit/` e deixar registrado que ela está vazia.
- [ ] **T20.** Servir `web/` localmente e percorrer a página inteira: hero, índice, quatro gráficos, troca de fluxo na seção 02, grade e rodapé.

## Ordem

T1–T5 primeiro. Sem o JSON, a página não tem o que mostrar. T6–T9 em seguida, para a casca existir. T10–T14 na ordem das seções. T15–T17 fecham a página. T18–T20 por último.

## Pronto quando

A página abre no navegador, os três números do hero batem com a tabela de conferência, o índice marca a seção visível, o ranking troca de fluxo sem recarregar, e o README descreve esse fluxo.
