# data/

Put the crystal structure to be analysed here as a `.cif`, and point
`crystal.cif` in `config.yaml` at it.

`2161927.cif` is the shipped example: AQx-2, CCDC entry 2161927, from
*Nat. Commun.* **14**, 5079 (2023). It is redistributed here for convenience;
the CCDC's terms apply, and the authoritative copy is the CCDC entry itself.

The toolkit expects an ordinary single-crystal CIF containing:

* `_cell_length_a/b/c` and `_cell_angle_alpha/beta/gamma`
* a `_space_group_symop_operation_xyz` loop
* an atom site loop with `_atom_site_label`, `_atom_site_type_symbol` and
  `_atom_site_fract_x/y/z`

`_atom_site_disorder_group` is used when present and ignored when absent.
Solvent and minor disorder components are removed automatically; see
`crystal.selection` in `config.yaml` if you need to override that.
