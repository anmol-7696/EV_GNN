import json
import pandas as pd
import folium
import random


def mappa_osservazioni_csv(
        percorso_csv: str,
        file_html: str = "mappa_osservazioni.html",
        zoom_start: int = 12,
        usa_satellite: bool = True,
        disegna_linea: bool = False
):
    """
    Carica il CSV e disegna, per ogni riga, i due punti di osservazione con lo stesso colore.

    Parametri
    ----------
    percorso_csv : str
        Percorso al file .csv.
    file_html : str, opzionale
        Nome del file HTML di output (default 'mappa_osservazioni.html').
    zoom_start : int, opzionale
        Livello di zoom iniziale (default 12).
    usa_satellite : bool, opzionale
        Se True utilizza lo sfondo satellitare Esri World Imagery; altrimenti OpenStreetMap.
    disegna_linea : bool, opzionale
        Se True collega con una polilinea le due osservazioni di ogni riga.

    Ritorna
    -------
    folium.Map
        Oggetto mappa che può essere ulteriormente personalizzato.
    """
    # --- Caricamento dati ----------------------------------------------------
    df = pd.read_csv(percorso_csv)

    # --- Parsing coordinate --------------------------------------------------
    # Ci si assicura che le stringhe siano nel formato "latitudine, longitudine"
    def parse_coord(coord_str):
        lat_str, lon_str = map(str.strip, coord_str.split(','))
        return float(lat_str), float(lon_str)

    coords1 = df['Observation 1 Coordinates'].apply(parse_coord)
    coords2 = df['Observation 2 Coordinates'].apply(parse_coord)

    # --- Centroide iniziale --------------------------------------------------
    # Media delle prime coordinate per centrare la mappa
    lat0, lon0 = coords1.iloc[0]
    mappa = folium.Map(location=[lat0, lon0],
                       zoom_start=zoom_start,
                       tiles=None)  # nessun layer di default

    # Layer satellitare o standard
    if usa_satellite:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr=("Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, "
                  "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, "
                  "and the GIS User Community"),
            name="Esri World Imagery",
            overlay=False,
            control=True
        ).add_to(mappa)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mappa)

    # --- Funzione per generare colori casuali --------------------------------
    def colore_random() -> str:
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # --- Aggiunta marker (e polilinee) ---------------------------------------
    for (lat1, lon1), (lat2, lon2) in zip(coords1, coords2):
        colore = colore_random()

        folium.CircleMarker(
            location=[lat1, lon1],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)

        folium.CircleMarker(
            location=[lat2, lon2],
            radius=6,
            color=colore,
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)

        if disegna_linea:
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=colore,
                weight=2,
                opacity=0.8
            ).add_to(mappa)

    # Add EV locations
    json_filename = '../../data/ev/denmark/DenamarkEVstations.json'

    with open(json_filename) as f:
        content = json.load(f)
        print(content)

    list_of_EV_stations = list()
    for elem in content:
        lat = elem['AddressInfo']['Latitude']
        long = elem['AddressInfo']['Longitude']
        list_of_EV_stations.append((lat, long))

    for ev_station in list_of_EV_stations:
        folium.CircleMarker(
            location=[ev_station[0], ev_station[1]],
            radius=6,
            color='#000099',
            fill=True,
            fill_opacity=0.9
        ).add_to(mappa)


    folium.LayerControl().add_to(mappa)
    mappa.save(file_html)
    print(f"Mappa salvata in '{file_html}'")
    return mappa

if __name__ == '__main__':
    # df = pd.read_csv('denmark/observation_traffic_metadata.csv', sep=',')
    # print(df.columns)
    # Assicurati di avere: pip install pandas folium
    mappa_osservazioni_csv('denmark/observation_traffic_metadata.csv',
                           file_html="osservazioni2.html",
                           zoom_start=9, usa_satellite=True, disegna_linea=True)
