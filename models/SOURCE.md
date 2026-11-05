# burst_gene_expression: source and constants

Model structure: the two-state ("telegraph") bursting gene-expression model
with mRNA and protein layers.

Citations:
- Peccoud J, Ycart B (1995) Markovian modeling of gene-product synthesis.
  Theor Popul Biol 48:222-234 (telegraph model).
- Raj A, Peskin CS, Tranchina D, Vargas DY, Tyagi S (2006) Stochastic mRNA
  synthesis in mammalian cells. PLoS Biol 4(10):e309.
- Shahrezaei V, Swain PS (2008) Analytical distributions for stochastic gene
  expression. PNAS 105:17256-17261 (promoter/mRNA/protein three-stage model).

ADAPTED, not copied. The papers give model structure and the qualitative
regime (slow promoter switching, bursty mRNA with a few copies, protein
with a long lifetime and hundreds of copies). The numeric constants below are
chosen by us to sit in that regime, in units of 1/min; they are not fitted
values from any one paper.

| id | value (1/min) | meaning | partition |
|----|---------------|---------|-----------|
| k_on  | 0.05 | promoter activation G_off -> G_on | stochastic |
| k_off | 0.15 | promoter deactivation G_on -> G_off | stochastic |
| k_tx  | 2.0  | transcription from G_on (per promoter) | stochastic |
| k_dm  | 0.1  | mRNA degradation (half-life ~7 min) | stochastic |
| k_tl  | 2.0  | translation (per mRNA) | continuous |
| k_dp  | 0.02 | protein degradation (half-life ~35 min) | continuous |

Initial state (copy numbers): G_off=1, G_on=0, mRNA=0, protein=0.

Expected regime: P(on)=k_on/(k_on+k_off)=0.25; mean mRNA = k_tx*0.25/k_dm = 5;
mean protein = k_tl*5/k_dp = 500.

All rate laws are mass-action (constant * species); one compartment of size 1,
species are amounts.
