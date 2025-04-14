package org.example;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.search.EntitySearcher;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            System.err.println("Usage: java -jar millm-owlapi-4.jar <folder_path> <level: s|v>");
            return;
        }

        String folderPath = args[0];
        String level = args[1].toLowerCase();

        if (!level.equals("s") && !level.equals("v")) {
            System.err.println("Error: level must be 's' for summary or 'v' for verbose");
            return;
        }

        File folder = new File(folderPath);
        File[] files = folder.listFiles((dir, name) -> name.endsWith(".owl") || name.endsWith(".xml"));

        if (files == null) {
            System.err.println("No OWL files found in folder.");
            return;
        }

        String outputFile = "../../data-processing/ontology_classes_" + (level.equals("v") ? "verbose.csv" : "summary.csv");
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();

        for (File file : files) {
            try {
                OWLOntology ontology = manager.loadOntologyFromOntologyDocument(file);
                for (OWLClass cls : ontology.getClassesInSignature()) {
                    String fileBase = file.getName().replaceAll("\\.owl|\\.xml", "");
                    String className = getClassFragment(cls);
                    String hyphenatedName = fileBase + "-" + className;

                    String verbose = buildVerboseClassRepresentation(cls, ontology);
                    String summary = buildSummaryClassRepresentation(cls, ontology);
                    String cleanedVerbose = verbose.replaceAll("<[^#>]*[#/]([^>]+)>", "$1").replaceAll("_", " ");

                    if (level.equals("v")) {
                        writer.write(escapeCsv(hyphenatedName) + "," +
                                escapeCsv(cleanedVerbose.replaceAll("_", " ")) + "\n");
                    }
                    else {
                        writer.write(escapeCsv(hyphenatedName) + "," +
                                escapeCsv(summary.replaceAll("_", " ")) + "\n");
                    }
                }
                manager.removeOntology(ontology);
            } catch (Exception e) {
                System.err.println("Failed to process " + file.getName() + ": " + e.getMessage());
            }
        }

        writer.close();
        System.out.println("Done. Output written to ontology_classes.csv");
    }
    private static String buildVerboseClassRepresentation(OWLClass cls, OWLOntology ontology) {
        String subject = getClassFragment(cls);
        StringBuilder sb = new StringBuilder();
        List<String> parts = new ArrayList<>();

        // SubClassOf
        for (OWLSubClassOfAxiom subAxiom : ontology.getSubClassAxiomsForSubClass(cls)) {
            String rendered = renderClassExpression(subAxiom.getSuperClass());
            if (rendered.startsWith("not (")) {
                rendered = "not " + rendered.substring(5, rendered.length() - 1).replaceAll("_", " "); // strip outer brackets
            }
            parts.add("is a SubClassOf " + rendered);
        }

        // SuperClassOf
        for (OWLSubClassOfAxiom superAxiom : ontology.getSubClassAxiomsForSuperClass(cls)) {
            parts.add("is a SuperClassOf " + renderClassExpression(superAxiom.getSubClass()).replaceAll("_", " "));
        }

        // Object properties where cls is domain
        for (OWLObjectPropertyExpression prop : ontology.getObjectPropertiesInSignature()) {
            if (!prop.isAnonymous()) {
                OWLObjectProperty namedProp = prop.asOWLObjectProperty();
                for (OWLObjectPropertyDomainAxiom domainAxiom : ontology.getObjectPropertyDomainAxioms(namedProp)) {
                    if (domainAxiom.getDomain().equals(cls)) {
                        for (OWLClassExpression range : EntitySearcher.getRanges(namedProp, ontology).collect(Collectors.toList())) {
                            parts.add(getClassFragment(namedProp).replaceAll("_", " ") + " " + renderClassExpression(range).replaceAll("_", " "));
                        }
                    }
                }
            }
        }

        sb.append(subject);
        if (!parts.isEmpty()) {
            sb.append(" ").append(String.join(" and ", parts));
        }

        return cleanDescriptionText(sb.toString().replaceAll("_", " "));
    }

    private static String buildSummaryClassRepresentation(OWLClass cls, OWLOntology ontology) {
        Set<String> related = new LinkedHashSet<>();

        for (OWLSubClassOfAxiom subAxiom : ontology.getSubClassAxiomsForSubClass(cls)) {
            related.add(renderClassExpression(subAxiom.getSuperClass()).replaceAll("_", " "));
        }

        for (OWLSubClassOfAxiom superAxiom : ontology.getSubClassAxiomsForSuperClass(cls)) {
            related.add(renderClassExpression(superAxiom.getSubClass()).replaceAll("_", " "));
        }

        for (OWLObjectPropertyExpression prop : ontology.getObjectPropertiesInSignature()) {
            for (OWLObjectPropertyDomainAxiom domainAxiom : ontology.getObjectPropertyDomainAxioms(prop)) {
                if (domainAxiom.getDomain().equals(cls)) {
                    for (OWLClassExpression range : EntitySearcher.getRanges(prop, ontology).collect(Collectors.toList())) {
                        related.add(renderClassExpression(range).replaceAll("_", " "));
                    }
                }
            }
        }

        // Class name + all related names, space-separated
        return getClassFragment(cls) + " " + String.join(" ", related);
    }

    private static String getClassFragment(OWLClass cls) {
        return getClassFragment((OWLEntity) cls);
    }

    private static String getClassFragment(OWLClassExpression expr) {
        if (!expr.isAnonymous()) {
            return getClassFragment(expr.asOWLClass());
        }
        return expr.toString();
    }

    private static String getClassFragment(OWLEntity entity) {
        String fragment = entity.getIRI().getFragment();
        return fragment != null ? fragment : entity.getIRI().toString();
    }

    private static String escapeCsv(String s) {
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    private static String cleanDescriptionText(String input) {
        // Replace DataExactCardinality(n <IRI> ...)
        input = input.replaceAll("DataExactCardinality\\((\\d+) <[^#>]+#([^>]+)> [^)]+\\)", "ExactCardinality $1 of $2");

        // Replace SubClassOf ObjectUnionOf(<IRI1> <IRI2> ...)
        Pattern unionPattern = Pattern.compile("SubClassOf ObjectUnionOf\\(([^)]+)\\)");
        Matcher matcher = unionPattern.matcher(input);
        StringBuffer buffer = new StringBuffer();

        while (matcher.find()) {
            String group = matcher.group(1);
            String[] iris = group.split(" ");
            List<String> classNames = new ArrayList<>();

            for (String iri : iris) {
                String name = iri.replaceAll(".*[#/](\\w+)>", "$1");
                classNames.add(name);
            }

            String replacement = "subclass of " + String.join(" and ", classNames);
            matcher.appendReplacement(buffer, Matcher.quoteReplacement(replacement));
        }
        matcher.appendTail(buffer);
        input = buffer.toString();

        return input;
    }

    private static String getObjectPropertyName(OWLObjectPropertyExpression prop) {
        if (!prop.isAnonymous()) {
            return getClassFragment(prop.asOWLObjectProperty());
        }
        return "anonymousObjectProperty";
    }

    private static String getDataPropertyName(OWLDataProperty prop) {
        return getClassFragment(prop);
    }

    private static String renderClassExpression(OWLClassExpression expr) {
        if (!expr.isAnonymous()) {
            return getClassFragment(expr.asOWLClass());
        }

        if (expr instanceof OWLObjectSomeValuesFrom) {
            OWLObjectSomeValuesFrom some = (OWLObjectSomeValuesFrom) expr;
            String prop = getObjectPropertyName(some.getProperty());
            String filler = renderClassExpression(some.getFiller());
            return "some " + prop + " " + filler;
        }

        if (expr instanceof OWLObjectAllValuesFrom) {
            OWLObjectAllValuesFrom all = (OWLObjectAllValuesFrom) expr;
            String prop = getObjectPropertyName(all.getProperty());
            String filler = renderClassExpression(all.getFiller());
            return "only " + prop + " " + filler;
        }

        if (expr instanceof OWLObjectExactCardinality) {
            OWLObjectExactCardinality card = (OWLObjectExactCardinality) expr;
            String prop = getObjectPropertyName(card.getProperty());
            return "exactly " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLObjectMinCardinality) {
            OWLObjectMinCardinality card = (OWLObjectMinCardinality) expr;
            String prop = getObjectPropertyName(card.getProperty());
            return "at least " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLObjectMaxCardinality) {
            OWLObjectMaxCardinality card = (OWLObjectMaxCardinality) expr;
            String prop = getObjectPropertyName(card.getProperty());
            return "at most " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLDataExactCardinality) {
            OWLDataExactCardinality card = (OWLDataExactCardinality) expr;
            OWLDataPropertyExpression propExpr = card.getProperty();
            String prop = (propExpr instanceof OWLDataProperty)
                    ? getDataPropertyName((OWLDataProperty) propExpr)
                    : "anonymousDataProperty";
            return "exactly " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLDataMinCardinality) {
            OWLDataMinCardinality card = (OWLDataMinCardinality) expr;
            OWLDataPropertyExpression propExpr = card.getProperty();
            String prop = (propExpr instanceof OWLDataProperty)
                    ? getDataPropertyName((OWLDataProperty) propExpr)
                    : "anonymousDataProperty";
            return "at least " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLDataHasValue) {
            OWLDataHasValue val = (OWLDataHasValue) expr;
            OWLDataPropertyExpression propExpr = val.getProperty();
            String prop = (propExpr instanceof OWLDataProperty)
                    ? getDataPropertyName((OWLDataProperty) propExpr)
                    : "anonymousDataProperty";
            return "has value " + val.getFiller().getLiteral() + " for " + prop;
        }

        if (expr instanceof OWLObjectUnionOf) {
            OWLObjectUnionOf union = (OWLObjectUnionOf) expr;
            return union.operands()
                    .map(Main::renderClassExpression)
                    .collect(Collectors.joining(" or "));
        }

        if (expr instanceof OWLDataMaxCardinality) {
            OWLDataMaxCardinality card = (OWLDataMaxCardinality) expr;
            OWLDataPropertyExpression propExpr = card.getProperty();
            String prop = (propExpr instanceof OWLDataProperty)
                    ? getDataPropertyName((OWLDataProperty) propExpr)
                    : "anonymousDataProperty";
            return "at most " + card.getCardinality() + " of " + prop;
        }

        if (expr instanceof OWLObjectHasValue) {
            OWLObjectHasValue val = (OWLObjectHasValue) expr;
            String prop = getObjectPropertyName(val.getProperty());
            String filler = val.getFiller().toStringID(); // Get the IRI fragment if possible
            filler = filler.replaceAll(".+[#/](\\w+)$", "$1"); // Extract fragment
            return "has value " + filler + " for " + prop;
        }

        if (expr instanceof OWLDataSomeValuesFrom) {
            OWLDataSomeValuesFrom some = (OWLDataSomeValuesFrom) expr;
            OWLDataPropertyExpression propExpr = some.getProperty();
            String prop = (propExpr instanceof OWLDataProperty)
                    ? getDataPropertyName((OWLDataProperty) propExpr)
                    : "anonymousDataProperty";
            String datatype = some.getFiller().toString();
            datatype = datatype.replaceAll(".*[#/](\\w+)>?$", "$1"); // extract xsd type
            return "some " + prop + " of type " + datatype;
        }

        if (expr instanceof OWLObjectComplementOf) {
            OWLObjectComplementOf comp = (OWLObjectComplementOf) expr;
            OWLClassExpression operand = comp.getOperand();

            // Special case: not (some P C) → does not relate to P C
            if (operand instanceof OWLObjectSomeValuesFrom) {
                OWLObjectSomeValuesFrom some = (OWLObjectSomeValuesFrom) operand;
                String prop = getObjectPropertyName(some.getProperty());
                String filler = renderClassExpression(some.getFiller());
                return "does not relate to " + prop + " " + filler;
            }

            // Default fallback: plain negation
            String inner = renderClassExpression(operand);
            return "not " + inner;
        }

        if (expr instanceof OWLObjectIntersectionOf) {
            OWLObjectIntersectionOf intersection = (OWLObjectIntersectionOf) expr;
            return "both " + intersection.operands()
                    .map(Main::renderClassExpression)
                    .collect(Collectors.joining(" and "));
        }

        if (expr instanceof OWLObjectOneOf) {
            OWLObjectOneOf oneOf = (OWLObjectOneOf) expr;
            List<String> individuals = oneOf.individuals()
                    .map(ind -> {
                        String iri = ind.toStringID();
                        return iri.replaceAll(".*[#/](\\w+)", "$1");
                    })
                    .collect(Collectors.toList());

            if (individuals.size() == 1) {
                return "one of " + individuals.get(0);
            } else if (individuals.size() == 2) {
                return "one of " + individuals.get(0) + " or " + individuals.get(1);
            } else {
                // Oxford comma for clarity
                String last = individuals.remove(individuals.size() - 1);
                return "one of " + String.join(", ", individuals) + ", or " + last;
            }
        }

        return expr.toString().replaceAll("<[^#>]*[#/]([^>]+)>", "$1").replaceAll("_", " "); // fallback
    }

}