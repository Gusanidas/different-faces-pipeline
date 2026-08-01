"""The two locked looks used throughout the visual case study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Demographic:
    lock: str
    occupations: tuple[str, ...]
    personal_details: tuple[str, ...]


DEMOGRAPHICS = {
    "es_m": Demographic(
        lock="a Spanish man in his early 30s, short medium-brown hair, tanned skin",
        occupations=(
            "works as a management consultant",
            "works night shifts as a hospital nurse",
            "teaches history at a public school",
            "manages his family's bakery",
            "drives a city bus",
            "works as a laboratory technician",
            "repairs elevators for a living",
            "works as a court interpreter",
            "runs a small neighborhood bookshop",
            "works as a bicycle mechanic",
            "delivers mail on a rural route",
            "works as a museum security guard",
            "works as a landscape architect",
            "manages a busy restaurant kitchen",
            "works as a coastal park ranger",
            "does accounting for a family business",
        ),
        personal_details=(
            "plays piano in his free time",
            "goes rock climbing on weekends",
            "restores vintage motorcycles",
            "plays chess at a neighborhood club",
            "sings in a local choir",
            "grows vegetables in a community garden",
            "collects and repairs old radios",
            "swims before work every morning",
            "coaches a youth football team",
            "sketches street scenes after work",
            "studies astronomy with an amateur group",
            "takes salsa classes twice a week",
            "volunteers at an animal shelter",
            "writes short stories on the train",
            "photographs old railway stations",
            "is afraid of heights",
        ),
    ),
    "sc_f": Demographic(
        lock="a Scandinavian woman in her late 20s, long light-blonde hair, fair skin, blue eyes",
        occupations=(
            "works as a software developer",
            "works night shifts as a midwife",
            "teaches science at a public school",
            "manages a family-owned cafe",
            "drives a city tram",
            "works as a marine laboratory technician",
            "repairs bicycles for a living",
            "works as a conference interpreter",
            "runs a small neighborhood bookshop",
            "works as a carpenter",
            "delivers mail on a rural route",
            "works as a museum conservator",
            "works as a landscape architect",
            "manages a busy restaurant kitchen",
            "works as a mountain park ranger",
            "does accounting for a family business",
        ),
        personal_details=(
            "plays cello in an amateur orchestra",
            "goes cross-country skiing on weekends",
            "restores wooden furniture",
            "plays chess at a neighborhood club",
            "sings in a local choir",
            "grows herbs in a community garden",
            "collects and repairs old cameras",
            "swims before work every morning",
            "coaches a youth handball team",
            "sketches street scenes after work",
            "studies astronomy with an amateur group",
            "takes contemporary dance classes",
            "volunteers at an animal shelter",
            "writes short stories on the train",
            "photographs old railway stations",
            "is afraid of heights",
        ),
    ),
}
