# Surface-runoff weekly LSTM with eight RiTHyMs static attributes

Controlled experiment: append eight static descriptors to the surface-runoff/elevation baseline. No GNN, attention or CAMELS. All original dynamic inputs, masks, targets, splits, normalization statistics and history remain unchanged. Only new static inputs and their training-site normalization are added.

## Comparison on identical valid test weeks

| Scope | Total-runoff median NSE | Surface-only median NSE | Surface + eight statics median NSE | Sites improved vs surface |
|---|---:|---:|---:|---:|
| all51 | 0.8209 | 0.7570 | 0.7473 | 26/51 |
| headwater10 | 0.8742 | 0.7970 | 0.8034 | 5/10 |

| Site | Valid weeks | Total NSE | Surface NSE | Surface + statics NSE | Change vs surface | New volume bias |
|---|---:|---:|---:|---:|---:|---:|
| ELF01 Douglas | 340 | 0.8883 | 0.8340 | 0.8212 | -0.0128 | -4.29% |
| ELF02 Sevierville | 340 | 0.8209 | 0.7376 | 0.7358 | -0.0017 | -13.90% |
| ELF03 South Holston | 337 | 0.8228 | 0.7829 | 0.7809 | -0.0019 | +8.27% |
| ELF04 Watauga | 332 | 0.8444 | 0.7971 | 0.8058 | +0.0087 | -3.60% |
| ELF05 Elizabethton | 340 | 0.4200 | 0.4382 | 0.4367 | -0.0015 | -30.01% |
| ELF06 Boone | 226 | 0.8200 | 0.7609 | 0.7145 | -0.0464 | +31.30% |
| ELF07 Kingsport | 226 | 0.8024 | 0.7309 | 0.7091 | -0.0218 | +24.72% |
| ELF08 Cherokee | 337 | 0.8515 | 0.8034 | 0.8061 | +0.0027 | +5.97% |
| ELF09 Maryville | 340 | 0.8571 | 0.7711 | 0.7555 | -0.0155 | +4.06% |
| ELF10 Fontana | 340 | 0.9339 | 0.8318 | 0.8459 | +0.0141 | -11.00% |
| ELF11 Chilhowee | 210 | -0.3434 | -0.2438 | -0.2506 | -0.0068 | -47.22% |
| ELF12 Tellico | 192 | 0.7700 | 0.7451 | 0.7473 | +0.0022 | +7.65% |
| ELF13 Norris | 340 | 0.8268 | 0.7823 | 0.7854 | +0.0031 | -5.38% |
| ELF14 Clinton | 304 | 0.5261 | 0.4799 | 0.4661 | -0.0138 | -0.67% |
| ELF15 Melton Hill | 304 | 0.5307 | 0.4899 | 0.4945 | +0.0047 | -6.77% |
| ELF16 Oakdale | 340 | 0.8656 | 0.7761 | 0.7751 | -0.0011 | -0.08% |
| ELF17 Chatuge | 338 | 0.8840 | 0.8137 | 0.8244 | +0.0107 | -16.67% |
| ELF18 Nottely | 340 | 0.8650 | 0.7570 | 0.7310 | -0.0260 | -18.89% |
| ELF19 Hiwassee | 340 | 0.8626 | 0.8039 | 0.8106 | +0.0066 | -8.64% |
| ELF20 Apalachia | 340 | 0.8857 | 0.7870 | 0.7884 | +0.0014 | -5.98% |
| ELF21 Blue Ridge | 340 | 0.9041 | 0.7836 | 0.7550 | -0.0285 | -13.70% |
| ELF22 Copperhill | 340 | 0.9033 | 0.8154 | 0.8141 | -0.0013 | -6.43% |
| ELF23 Ocoee No. 1 | 340 | 0.8773 | 0.7851 | 0.8099 | +0.0249 | +2.05% |
| ELF24 Charleston | 302 | 0.7016 | 0.5867 | 0.6111 | +0.0244 | -11.39% |
| ELF25 South Chickamauga Creek | 340 | 0.8586 | 0.7639 | 0.7719 | +0.0080 | +0.55% |
| ELF26 Whitwell | 340 | 0.8940 | 0.7593 | 0.7656 | +0.0063 | +13.19% |
| ELF27 Woodville | 340 | 0.8679 | 0.7759 | 0.7834 | +0.0075 | +8.55% |
| ELF28 Chase | 340 | 0.7995 | 0.7375 | 0.7427 | +0.0052 | -5.99% |
| ELF29 Tims Ford | 330 | 0.8834 | 0.7970 | 0.8069 | +0.0099 | +2.32% |
| ELF30 Fayetteville | 314 | 0.6866 | 0.6014 | 0.5935 | -0.0079 | +43.09% |
| ELF31 Prospect | 324 | 0.8168 | 0.7450 | 0.7383 | -0.0067 | +0.83% |
| ELF32 Iron City | 340 | 0.7363 | 0.6639 | 0.6691 | +0.0052 | -6.77% |
| ELF33 Upper Bear Creek | 311 | 0.8624 | 0.8213 | 0.8291 | +0.0078 | +3.97% |
| ELF34 Bear Creek | 238 | 0.6516 | 0.6192 | 0.6000 | -0.0192 | +61.84% |
| ELF35 Little Bear Creek | 237 | 0.7878 | 0.7693 | 0.7568 | -0.0126 | +12.62% |
| ELF36 Cedar Creek | 296 | 0.8263 | 0.7825 | 0.7782 | -0.0043 | +15.01% |
| ELF37 Bishop | 159 | 0.6772 | 0.6480 | 0.6410 | -0.0070 | +18.82% |
| ELF38 Lobelville | 340 | 0.6936 | 0.5396 | 0.5398 | +0.0002 | +11.60% |
| ELF39 Normandy | 338 | 0.8596 | 0.8029 | 0.8010 | -0.0019 | +10.68% |
| ELF40 Shelbyville | 337 | 0.8134 | 0.7568 | 0.7534 | -0.0034 | +12.47% |
| ELF41 Hurricane Mills | 340 | 0.6482 | 0.4735 | 0.4944 | +0.0209 | -5.21% |
| ELF42 Knoxville | 192 | 0.7148 | 0.6234 | 0.6298 | +0.0064 | +10.11% |
| ELF43 Fort Loudoun | 192 | 0.7599 | 0.6912 | 0.6909 | -0.0003 | +15.13% |
| ELF44 Watts Bar | 204 | 0.8252 | 0.7094 | 0.7124 | +0.0030 | +16.88% |
| ELF45 Chickamauga | 306 | 0.8552 | 0.7683 | 0.7778 | +0.0095 | +11.17% |
| ELF47 Guntersville | 272 | 0.7802 | 0.6060 | 0.5838 | -0.0222 | +56.29% |
| ELF48 Decatur | 301 | 0.8385 | 0.7808 | 0.7952 | +0.0144 | -5.59% |
| ELF49 Wheeler | 289 | 0.6778 | 0.6112 | 0.6406 | +0.0294 | -17.40% |
| ELF50 Wilson | 159 | 0.7206 | 0.5901 | 0.6192 | +0.0291 | +20.03% |
| ELF51 Pickwick | 159 | 0.7580 | 0.6627 | 0.6553 | -0.0074 | +10.72% |
| ELF52 Savannah | 159 | 0.2374 | 0.3740 | 0.2906 | -0.0834 | +25.48% |

## Inputs and processing

Dynamic: weekly P amount, weekly mean T, weekly NLDAS surface runoff Qs. Original static elevation plus eight descriptors: soil permeability, sand %, clay %, depth to bedrock, depth to water table, baseflow index %, 2019 riparian impervious %, 2019 open-water %. Soil depth/permeability source-native units retained without unverified conversion.
Raw local-catchment attributes are aggregated using existing ELF forcing-region intersection areas (EPSG5070). Missing catchment values are excluded and valid area weights renormalized; no values are filled. The impervious metric describes 100m riparian buffers, not exact basin-wide impervious cover. Detailed field mapping, region coverage and provenance are in static_audit.json and the experiment static_preparation folder.
2019 land cover is fixed by user request for future use. Retrospective 2010–2016 evaluation uses these post-period descriptors and is not a strict as-of-date hindcast. Static BFI is from external StreamCat, not calculated from ELF test targets.

## Training settings

52 sites trained; 51 scoreable in testing. Train1982–2004; validation2005–2009; test2010–October2016. Target week plus 12 preceding weeks, not a 13-week forecast lead. 50 epochs, no early stopping; validation loss on the ten priority headwaters selects the checkpoint.
Shared one-layer LSTM hidden64; dropout0.2 before 52 independent nonlinear heads64→16 ReLU→1 Softplus. 74,932 parameters. Fresh seed42; original parameters initialized identically to surface-only baseline, new static input columns start at zero. Adam lr0.001, batch512, clip1, cosineT_max50 eta_min0.00001. Same normalized squared-error loss.
Selected epoch 20; training/validation time 30.8 seconds.

## Interpretation

This is a single-seed historical-input experiment. The extra statics can describe basin differences but do not provide time-varying baseflow. Actual HEADWATER forecast skill remains untested. Test outcomes do not select the checkpoint.

Full metric summary:

```json
{
  "all51": {
    "locations": 51,
    "paired_test_weeks": 15025,
    "improved_nse_vs_surface": 26,
    "improved_nse_vs_total": 3,
    "median_nse_surface": 0.7569858461546357,
    "median_kge_surface": 0.792442957074564,
    "median_abs_bias_surface": 10.16471408220937,
    "median_nse_static8": 0.7472752419234157,
    "median_kge_static8": 0.7917224697841021,
    "median_abs_bias_static8": 10.722742368137284,
    "median_nse_total": 0.8209257062192478,
    "median_kge_total": 0.830906985711308,
    "median_abs_bias_total": 7.141298940729346
  },
  "headwater10": {
    "locations": 10,
    "paired_test_weeks": 3375,
    "improved_nse_vs_surface": 5,
    "improved_nse_vs_total": 0,
    "median_nse_surface": 0.7970495985733776,
    "median_kge_surface": 0.817868773242318,
    "median_abs_bias_surface": 7.4627709031341585,
    "median_nse_static8": 0.8034413423700285,
    "median_kge_static8": 0.8237526645096932,
    "median_abs_bias_static8": 9.474677037130597,
    "median_nse_total": 0.8741981822262809,
    "median_kge_total": 0.8887923960033592,
    "median_abs_bias_total": 6.294699092747875
  }
}
```
