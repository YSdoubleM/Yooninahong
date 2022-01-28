import detect

opt = detect.parse_opt('--weight .\best_0126.pt --img 320 --conf 0.4 --source .\Average_stature__out__Market__00003__scene00481.png')
detect.main(opt)