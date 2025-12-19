def main():
    
    args = parse_arguments()
    
    # Set logging level
    if args.verbose: logger.setLevel('DEBUG')
    
    try:
        df = LoadData(StockList)

        kms = KMeans(df)
        ksm_report = kms.run()

        gmm = GMM(df)
        gmm_report = gmm.run()

        hdb = HDBScan(df)
        hdb_report = hdb.run()

        reportAll(kms_report, gmm_report, hdb_report)

    except KeyboardInterrupt:
        logger.info("Analysis interrupted")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
