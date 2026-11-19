# Input data format

Geneformer consumes single-cell transcriptomes as **rank value encodings**, produced
by its `TranscriptomeTokenizer`. This project's `tokenize` stage runs that tokenizer
for you; you provide a `.loom` or `.h5ad` file (format inferred from the suffix).

## Recommended: loom

The synthetic sample data and the smoke test use **loom**, and it is the recommended
format for this pinned Geneformer version. Required attributes:

| Location (loom)     | Attribute     | Meaning                                      |
|---------------------|---------------|----------------------------------------------|
| row attribute (`ra`)| `ensembl_id`  | Ensembl gene ID per gene                     |
| column attr (`ca`)  | `n_counts`    | Total counts per cell (for normalization)    |
| column attr (`ca`)  | your label    | e.g. `cell_type`, `disease` — set in config  |
| matrix              | —             | Raw counts, shape **genes x cells**          |

## h5ad caveat

`.h5ad` is also accepted (`ensembl_id` in `.var`, `n_counts` in `.obs`, labels in
`.obs`), **but** this Geneformer version's h5ad tokenization path indexes a pandas
Series positionally and raises `KeyError` when AnnData coerces the `.var` index to
strings (which it does by default). Loom avoids this entirely by using NumPy
positional indexing. If you must use h5ad, ensure `.var` has a true integer
`RangeIndex` at tokenization time, or convert to loom first.

## Example

See `src/scfm_ft/data.py` / `scfm-ft make-sample-data` for a minimal
correctly-shaped AnnData. For a real cell-type-annotation dataset, 10x PBMC3k and the
Hao 2021 CITE-seq reference are common choices (processing examples exist in the
sibling project `../fine-tune-single-cell-rna-seq-fm/scripts/processing`).

## Cell vs gene classification

- **Cell classifier** (`classifier.kind: cell`): labels live in `adata.obs`; set
  `label_column` and optionally `states` to restrict to specific label values.
- **Gene classifier** (`classifier.kind: gene`): instead of obs labels, you provide a
  `{Gene_label: [ensembl_ids...]}` mapping. This project currently wires the cell
  path end to end; the gene path uses the same `Classifier` and can be added by
  passing `gene_class_dict` in `finetune.py`.
