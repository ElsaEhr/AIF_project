from retrieval import load_assets, discover_movies

if __name__ == "__main__":
    annoy_index, metadata, model, preprocess, device = load_assets()

    query = "a film with samurais and swords in Japan"
    results = discover_movies(query, 5, annoy_index, metadata, model, device)

    print("Top results:")
    for r in results:
        print(f"- id={r['id']} | category={r['category']} | dist={r['distance']:.3f} | poster={r['poster_path']}")
