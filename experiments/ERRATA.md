# Historical errata

## `direct_decode_probe.py`: covariance temperature was a no-op

The archived probe described configuration D as a “hot” covariance sample with temperature 1.3. The implementation multiplied the PCA coefficient vector by a uniform scalar and then immediately normalized the resulting deviation to a separately sampled shell radius:

```python
z = normal(...) * sqrt(lambda) * temperature
deviation = z @ components
deviation *= sampled_radius / norm(deviation)
```

The scalar temperature cancels exactly. Configurations A and D sampled the same distribution; their concrete vectors differed only because the shared random generator had advanced between calls. No A-versus-D result supports a covariance-temperature claim.

This does not affect the production result. Production diversity came from the explicitly wider p60–p99 shell used by configuration C and from cosine pre-gating. The cleaned `sample_cloud` API therefore exposes shell percentiles and no temperature parameter.
