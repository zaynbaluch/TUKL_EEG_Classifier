try:
    from mspca import mspca
    print("MSPCA Imported")
    print("Methods in mspca.MultiscalePCA:")
    print(dir(mspca.MultiscalePCA))
    print("\nHelp on fit_transform:")
    help(mspca.MultiscalePCA.fit_transform)
except Exception as e:
    print(f"Error: {e}")
