# === create_town_centroids.py ================================================
# 用途:
#  町丁の代表点（経緯度）CSVを作成します。
# 出力:
#  subject3/src/layer5/data/processed/town_centroids.csv
#
# 2通りの入力に対応:
#  A) 町丁ポリゴン(Shapefile/GeoJSON/Geopackage等)がある場合 → centroidsを計算してCSV化
#  B) 既に町丁ごとの経緯度が入ったCSVがある場合 → 必要列のみ抽出してCSV化
#
# 使い方（例）:
#   1) A) ポリゴンから作る場合:
#        python create_town_centroids.py --from-geo subject3/data/town_polygons.geojson --id-col town_id --name-col town
#      B) 既存CSVから作る場合（既に lon/lat 列あり）:
#        python create_town_centroids.py --from-csv subject3/data/town_points.csv --id-col town_id --lon-col lon --lat-col lat
#
# 生成物の想定カラム:
#   town_id, lon, lat
# ============================================================================

import argparse
import os
import sys
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def from_csv(csv_path, id_col, lon_col, lat_col, out_csv):
    df = pd.read_csv(csv_path)
    for c in (id_col, lon_col, lat_col):
        if c not in df.columns:
            sys.exit(f"[ERROR] '{csv_path}' に必要列が見つかりません: {c}")
    out = df[[id_col, lon_col, lat_col]].dropna().copy()
    out[id_col] = out[id_col].astype(str)
    out.columns = ["town_id", "lon", "lat"]
    out = out.drop_duplicates(subset=["town_id"])
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] W用のセントロイドCSVを書き出しました: {out_csv} (rows={len(out)})")

def from_geo(geo_path, id_col, name_col, out_csv):
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("[ERROR] geopandas が見つかりません。`pip install geopandas pyproj shapely` を実行してください。")
    gdf = gpd.read_file(geo_path)
    if id_col not in gdf.columns:
        sys.exit(f"[ERROR] '{geo_path}' にID列が見つかりません: {id_col}")
    if gdf.crs is None:
        sys.exit(f"[ERROR] '{geo_path}' のCRSが不明です。正しい座標参照系を設定してください。")
    # 経緯度(WGS84)に統一してから centroid → lon/lat を抽出
    gdf_ll = gdf.to_crs(epsg=4326).copy()
    cent = gdf_ll.geometry.centroid  # shapely
    out = pd.DataFrame({
        "town_id": gdf_ll[id_col].astype(str),
        "lon": cent.x,
        "lat": cent.y,
    })
    # （任意）名前も付けたいなら:
    if name_col and name_col in gdf_ll.columns:
        out["town"] = gdf_ll[name_col].astype(str)
    out = out.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["town_id"])
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] ポリゴンからセントロイドCSVを書き出しました: {out_csv} (rows={len(out)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-geo", type=str, default=None, help="町丁ポリゴンのパス（.shp/.geojson/.gpkg など）")
    parser.add_argument("--from-csv", type=str, default=None, help="既存の経緯度CSVのパス（lon/lat列を含む）")
    parser.add_argument("--id-col", type=str, default="town_id", help="町丁ID列名（入力側）")
    parser.add_argument("--name-col", type=str, default=None, help="町丁名列（任意）")
    parser.add_argument("--lon-col", type=str, default="lon", help="経度列（入力CSVの場合）")
    parser.add_argument("--lat-col", type=str, default="lat", help="緯度列（入力CSVの場合）")
    args = parser.parse_args()

    out_dir = os.path.join("subject3", "src", "layer5", "data", "processed")
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "town_centroids.csv")

    if args.from_geo and args.from_csv:
        sys.exit("[ERROR] --from-geo と --from-csv はどちらか片方だけ指定してください。")

    if args.from_geo:
        from_geo(args.from_geo, id_col=args.id_col, name_col=args.name_col, out_csv=out_csv)
    elif args.from_csv:
        from_csv(args.from_csv, id_col=args.id_col, lon_col=args.lon_col, lat_col=args.lat_col, out_csv=out_csv)
    else:
        sys.exit("[ERROR] 入力がありません。--from-geo か --from-csv を指定してください。")

if __name__ == "__main__":
    main()
