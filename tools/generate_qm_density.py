"""Optional fixed-geometry QM single point for an explicitly prepared fragment.

This does NOT automatically cap cut protein bonds or infer charge/spin. Prepare
the fragment chemically, including hydrogens/link atoms, before using this tool.
No large protein calculation is run automatically by training or sampling.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--check-dependencies', action='store_true')
    args = parser.parse_args()
    if importlib.util.find_spec('pyscf') is None:
        raise SystemExit('PySCF unavailable. Run on a supported QM environment or import externally calculated density CUBEs. No proxy was generated.')
    if args.check_dependencies:
        print('PySCF available')
        return
    if args.job is None:
        parser.error('--job is required')
    from pyscf import gto, dft, qmmm
    from pyscf.tools import cubegen
    job = json.loads(args.job.read_text(encoding='utf-8'))
    for key in ('atoms', 'charge', 'spin', 'basis', 'xc', 'output_cube', 'boundary_prepared'):
        if key not in job:
            raise ValueError(f'Missing explicit QM setting: {key}')
    if job['boundary_prepared'] is not True:
        raise ValueError('Prepare fragment hydrogens/link atoms and charge/spin before SCF')
    if len(job['atoms']) > int(job.get('max_atoms', 500)):
        raise ValueError('Fragment exceeds explicit atom budget; review cost before increasing max_atoms')
    output = (args.job.parent / job['output_cube']).resolve()
    if output.exists():
        raise FileExistsError(f'Will not overwrite existing density: {output}')
    mol = gto.M(atom=job['atoms'], unit='Angstrom', charge=int(job['charge']),
                spin=int(job['spin']), basis=job['basis'], max_memory=int(job.get('memory_mb', 2000)))
    solver = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    solver.xc = job['xc']
    solver.max_cycle = int(job.get('max_cycle', 100))
    if 'mm_charges' in job:
        solver = qmmm.mm_charge(solver, job['mm_coords_A'], job['mm_charges'], unit='Angstrom')
    energy = solver.kernel()
    if not solver.converged:
        raise RuntimeError('SCF did not converge; no density label will be written')
    dm = solver.make_rdm1()
    if dm.ndim == 3:
        dm = dm.sum(axis=0)  # Total electron density, not spin density.
    spacing_A = float(job.get('grid_spacing_A', 0.25))
    margin_A = float(job.get('margin_A', 10.0))
    if spacing_A <= 0 or margin_A < 0:
        raise ValueError('Invalid CUBE grid spacing/margin')
    cubegen.density(mol, str(output), dm, resolution=spacing_A / 0.529177210903,
                    margin=margin_A / 0.529177210903)
    print(f'Converged E={energy:.12f} Hartree; total electron density in e/bohr^3: {output}')
    print('Import with an explicit frame/atom mapping; only sampled 10-A ROI enters supervision.')


if __name__ == '__main__':
    main()
