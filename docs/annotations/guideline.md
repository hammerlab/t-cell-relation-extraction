# Annotation Guidelines

Cytokines to remove: 
rules: Do something with "respectively"
ways to make process better:
  - Build NER models first
  - Decide upfront if recall should across all relations, or the simplest ones in a corpus
      - In other words, is it ok to simply ignore positive examples because they're more complex?
  - Use a sentence complexity filter as in iX prior to annotation as well as candidate filtering
  - Annotate candidates (not documents) with ability to expand certain cases into more examples (no such tool exists)
  - Start by doing minimal supervision until models beat baseline, then work on improving with snorkel
      - Error analysis on this should have been used to inform heuristics

## All Relations

- **Positive**
    - **Conditional Relations**: Relations that only occur under certain conditions should be considered valid
        - "Moreover, our results demonstrated that exposure to IL-12 and IL-18 induces robust **IFN-γ** expression by ex-vivo expanded **Vγ9Vδ2 T** cells"
            - IFN-γ expression by Vγ9Vδ2 T cells may not always occur, but relations should simply capture that the behavior has been observed
- **Negative**
    - **Unconfirmed Relations**: Any "unconfirmed" relationships should never be positive examples
        - "Evidence defining the role of IL-2 in TH1 differentiation is minimal"
    - **Ambiguous Relationships**: Any language that does not make it absolutely clear what the association is between cells, proteins, and a related behavior
        - "including IL-17 , IL-6 , IL-1β , TNF-α and IL-22 , which carry the inflammatory signature of RA and are crucial in the differentiation and maintenance of pathogenic Th17 cells"
            - In this case, it is not clear if all of the cytokines apply to both differentiation AND maintenance or if only some apply to one or the other
        - "Our data suggest that **T-bet and Eomes** regulate the balance between **TEM, TCM, and TSCM**"
            - When relations are expressed in many-to-many relationships rather than one-to-one, they should be made negative
        - "The spleens of calcitriol-treated mice had fewer splenic **Th17** cells and lower **IL-17** production than the placebo controls"
            - This implies that Th17 cells probably produce IL-17, but does not state so directly
        - "Our results demonstrated that expression of the molecules that regulate **IL-10** expression, such as Rorc and maf, was not affected by the lack of Trim33 in **Th17** cells"
            - This kind of transitive link does not directly implicate IL-10 expression by Th17 cells since it does not state in what way Rorc and maf regulate IL-10
        - "purified naive Th cells could simultaneously commit to the **Th1** and **Th2** differentiation programs by the integration of **IFN-γ** and **IL-4** signals"
            - It is not clear which cytokines are associated with induction of which cell type
        - "TCF-1 promotes the development of the **Th2** cell fate by promoting GATA-3 expression and increasing **IFN-γ** production"
            - IFN-γ is implied as conducive of Th2 differentiation but not strongly enough to make this a positive instance
        - "hosts also develop a number of regulatory mechanisms , including generating **Tregs** and production of **IL-10**"
            - This weak implication that IL-10 may be produced by Tregs does not constitute a definitive positive example (and should be considered negative)
        - "The data document a cellular and molecular link among **IL-10, IL-1**, and **Th17** cells"
            - It is not clear whether or not this "link" pertains to secretition or induction
        - "We concentrated on the **Th1 and Th2** effector cytokines **IFNγ and IL-4**"
            - While with some domain knowledge it is clear what the interpretation of this sentence is, this should be a negative example because the relations are not explicit
    - **Multiple References**: Relations should only be considered between the most obvious entities in a sentence (even if multiple references of the same entity occur)
        - "STAT4, when activated by **IL-12**, results in the development of Th1 cells and the production of the hallmark **Th1** cytokine IFNγ."
            - The relation exists between IL-12 and the first Th1 entity span but not the second
        - "while the expression of other **Th2** signature cytokines like Il5 , Il13 and the master Th2 transcription factor **Gata3** was decreased"
            - Again, only a candidate including the Th2 entity closest to the Gata3 entity would comprise a positive candidate (this one does not)
    - **Hypothesized Relations**: Relationships cannot be implied as in question as a part of the study, or in the details of what an assay is to measure
        - “we tested whether the role of **IL-2** in **TH1** differentiation”
        - “To clarify the role of **STAT5** in **TH1** differentiation, we ...”
        - "we specifically looked at the effect of IL23R in **Th17** cells, using **IL-17A** as read out"
            - This implies that IL-17A may be produced by Th17s, but does not confirm the results of that "read out"
        - "**Th1** and cytotoxic CMI responses to pH1N1 monovalent vaccine were characterized by the expression of **interleukin-2** ( IL-2 )"
            - This implies that expressed IL-2 is measured for Th1 cells, but not that expression was found
    - **Inhibitory Effects**: Anything relation implied as inhibitory in a general sense can be marked negative
        - "**IL-35** is also a suppressive cytokine known to have an inhibitory effect on lymphocyte proliferation and effector **Th17** of CD4+ CD25+ cells"
    - **Environment/Culture Presence**: A cytokine or TF that is simply "in the environment", with no further details, is not enough to imply a relation
        - "**CD4 cells** differentiated in the presence of **TGF-β**"
        - "when CD4 + T cells are activated under **Th9** culturing conditions (**TGF-β** and **IL-4**)"

## Individual Relations

#### Inducing Transcription Factor

- **Positive**
    - **TF Dependence**: A statement implying that the transcription factor is crucial to the function of a cell type, a part of "polarizing conditions", required for development, or able to regulate the "signature" cytokines
        - "JunB is critical for Il4 transcription, intimating a function for **JunB** in physiological **Th17** cell effector conversions"
        - "**TBX21** (also known as **T-bet**) is responsible for the **Th1** subtype"
        - "**cAMP** has been demonstrated to induce a **Th2** bias when present during T cell priming"
        - "... recently reported that ITK signals via **IRF4** to regulate naive cell differentiation to **Th9** cell fate"
        - "... cooperates with **BATF** to activate **Th17** signature genes"
        - "**T-bet** is a direct transcriptional regulator of **Th1** cytokines"
        - "**Foxo1** is induced in **Th9** cells, which in turn binds to both IL-9 and IRF-4 promoter thereby contributing to optimal expression of **IL-9 in Th9** cells"
            - Here, Foxo1 ultimately leads to expression of the signature cytokine, making this a positive example, but it would be negative if the relationship between Foxo1 and IL-9 was not provided
    - **TF Association**: A statement indicating that the TF is a part of the canonical definition of a cell type or otherwise strongly associated with its development/polarization (or may be used to identify the cell in an assay)
        - "... expression of **BCL6** in activated CD4 + T cells, which is associated with production of IL-21 and a **TFH** phenotype"
        - "**Helios** can be used to identify **Treg** cells"
        - "Transcripts enriched in Th17 versus Th1 were previously associated with **Th17** polarization (**RORγt, ARNTL, PTPN13, and RUNX1**) ."
        - "**TH1** cells, characterized by expression of the transcription factor **T-bet**"
    - **TF Possession**: A statement implying that the TF belongs to the cell type or is otherwise synonymous with it
        - "... T cells expressing the **Th17** transcription factor **RORγt**"
        - "mRNA expression levels of **RORγt** and STAT-3, transcriptional factors of **Th17** cells, were markedly higher"
        - "... while downregulating markers of Th1 and **Th2** cells (Tbet and **Gata3**, respectively)"
    - **Cell Type Specificity**: A TF that is indicated as being "specific" to a cell type
        - "c-Maf is a transcription factor specific for Th2 cells"
    - **Transcriptional Program**: Any TF that is in the "transcription program", if there is not an indication that the transcription program is explicitly not related to development (this language is generally used in the context of cell development, so it is OK if the "transcription program" referenced with no further qualification)
        - "relies on a common TH17 transcription programme composed of BATF, IRF4, STAT3 and RORγt"
    - **Constitutive Expression**: A TF that is expressed by the cell type, IF there is a further indication that the TF is a part of cell identity or integral in the differentiation process
        - "T regulatory cells (Tregs) that constitutively express the transcription factor (TF) FOXP3"
    - **Master Regulator**: Any TF that is a "master regulator" of a cell type
        - "**IRF4** acts as a master regulator of mucosal **Th17** cell differentiation"
        - "**Tregs** are a subset of T cells controlled by the master transcription factor **Foxp3**"
    - **Differentiation Inhibition/KO**: A relationship that implies that a cell type cannot form when the TF is inhibited or knocked out
        - "STAT5 is a key physiological inhibitor of Bcl6 expression and thereby an inhibitor of TFH cell differentiation" ==> (Bcl6 induces TFH)
        - "LXR inhibits **Th17** cell differentiation by interfering with the **aryl hydrocarbon receptor** mediated IL-17 transcription"
        - "the absence of **Foxo1** severely curtailed the development of Foxp3+ **regulatory T** cells"
        - "IL-2 is known to inhibit **Th17** differentiation by suppressing the expression of **RORγt**"
        - "The effect of **c-Rel** deficiency on **Th17** cell differentiation was even more dramatic"
- **Negative**
    - **Weak Association**: A relationship implied with no further information
        - RORγt is not required for the thymic development of T cells with the potential to form Th17 cells
    - **Explicit Expression**: Expression of a TF is not in itself enough to indicate that the TF is required for cell type formation unless the surrounding context also implies that the TF is, by itself, enough to be able to identify the cell type (in which case it can be a positive example)
        - "STAT5+ Th17 cells"
        - "Vδ2 + **γδT** cells express high levels of IL-18Rα and **PLZF**"
        - The most common form of this case, by far, is "FOXP3+ Treg" which is, by itelf, a NEGATIVE example.  A few concrete examples since this is such a common case:
            - This is a positive example since it appeals to how the TF induces or generates the cell type:
                - "simple activation of mouse T cells in the absence of **T regulatory** cell-inducing factor **FoxP3** is more tightly regulated"
            - These on the other hand, are ALL NEGATIVE cases:
                - "CD25 and folate receptor 4 (FR4) are constitutively expressed by **Foxp3**+ **Treg** cells"
                - "SOCS1‐deficient **Tregs** easily lose **Foxp3** expression and are converted into Th1‐ or Th17‐like effector cells"
    - **Inhibitory Relationship**: A TF that inhibits the function of a cell type (i.e. is not a "promoter" of the cell)
        - "BATF inhibits Th1 response"
        - "Dectin-1–mediated signals enhance **Th17** differentiation by inhibiting **T-bet** expression"
            - These may eventually be worth classifying separately, but for now should be negative examples


#### Inducing Cytokine

- **Positive**
    - **Cell Type Induction**: In the simplest case, a statement expressing that the cytokine induces, generates, skews, or polarizes the cell type differentiation directly
        - "together with IL-6, NO suppressed **Treg** development induced by **TGF-β** and retinoic acid"
        - "NO antagonized IL-6 to block **TGF-β**–directed **Th17** differentiation"
        - "We next tested whether **IFNγ**, which contributes to **Th1** differentiation"
        - "**IL-21**, a key cytokine produced by Tfh cells, promotes reactions in GCs that skews Tfr to **Tfh** cells"
    - **Cytokine Dependence**: A statement implying that a cytokine is necessary to generate a cell type
        - "Differentiation of Th2 cells is dependent on the availability of IL-4"
        - "For **Treg** induction, we also added **TGFβ**"
        - "differentiation of highly pathogenic **Th17** cells from naïve T cells occurs in the presence of **IL-23 , IL-6 , and TGF-β1**"
    - **Cell Type Identity**: A cytokine that leads to a cell type being identifiable through a particular phenotype
        - "it is known that IL-4 contributes to establishing Th2 identity in vivo"
        - **Note:** This is similar to the language that implies what a "signature" cytokine is for a cell type but it has an entirely different connotation; if the distinction is not clear in any one example, it should be made negative
- **Negative**
    - **Secreted Cytokine**: Any "Positive" rule for the secreted cytokine relation can be assumed as a disqualifing condition for this relation
    - **Explicit Inhibition**: A cytokine may be stated as eplicitly preventing differentiation of a specific cell type
        - "Research has shown that **IL6** is critical in preventing the conversion of naive Th cells into **Treg**"
        - "**IL-23** could antagonize development of Foxp3+ **Treg** cells, facilitating the development of intestinal inflammation"
    - **Response Promotion**: A cytokine that simply induces a "response" from a particular cell type
        - "Interleukin 23 (IL- 23) as a proinflammatory cytokine suppresses T regulatory cells (Treg) and promotes the response of T helper 17 (Th17) and T helper 1 (Th1) cells."
        - "the majority of the **Th1** TCCs are capable of responding to **IL-12** in a specific and consistent fashion"
        - "**IL-12** is a pro-inflammatory cytokine that enhances **Th1** responses by increasing IFN-γ"
    - **Precursor Cell Types**: The originating cell type (usually naive T cells) should not be implicated in relations
        - "**IL-6** was considered to promote the **naïve T** cells to differentiate into Th17 cells"
            - The IL-6 -> Th17 relation is there but not the IL-6 -> naive relation
    - **Cytokine Consumption**: Stating that a cell "consumes" or "reacts" with a cytokine is not enough to assume that it is involved in differentiation
        - "Metabolic disruption of effector T cells is mediated by **Treg** cell consumption of **IL-2**"
    - **Cell Dysfunction**: A cytokine that contributes to the "dysfunction" of a cell type
        - "**IL-17** is crucial in the differentiation and maintenance of pathogenic Th17 cells and dysfunction of **Treg** cells"
    - **Cell Proliferation/Activation**: Cytokines stated as involved in proliferation, activation, or maintenance can be ignored
        - "**IL-21** and **IL-23** activate STAT3 and are required for the maintenance of **iNKT** and **MAIT** cells"
    - **Cytokine Regulation**: Implying that a cytokine is regulated within a cell type is not sufficient evidence to determine what function that cytokine has
        - "**IL-4** is known to be extensively regulated by the action of various transcription factors and epigenetic mechanisms in **Th2** cells"
        - "Differential transcriptional regulation of **IL-10** in **Th1**"


#### Secreted Cytokine


- **Positive**
    - **Cytokine Secretion**: In the simplest case, a statement implying that the cell type releases, secretes, or generates the cytokine directly
        - "accompanied by the enhancement of the metabolic product of probiotics with the help of **IL-5** released by CD4+ **Th2** cells"
        - "flow cytometric analysis showed increased differentiation of miR-142a-5p-transfected cells towards **IFN-γ** producing **Th1** subtype"
        - "In 2008, one novel CD4+ T cell subset secreting high levels of **IL-9** was characterized and named **TH9** cells"
        - "The observed difference between **IL-13** reporter expression among thymic **iNKT** cells led us to ..."
    - **Cytokine Possession**: Any indication that a cytokine is a part of the "signature", "effector", or "cardinal" set of cytokines for a cell type or is otherwise very often associated with it, as long as no surrounding context implies that this it is fate-inducing
        - “The Th1 cytokine IL-12”
        - “Th-1-type cytokines such as interferon-γ (IFN-γ)”
        - “Th2 cytokines (IL-13)”
        - "along with decreased expression of **Th2** cytokines IL-4, IL-5 and IL-13"
        - "The signaling cascade of **IL-17**, the signature cytokine of **Th17** cells"
        - "The expression levels of the **Th17** effector cytokines IL17A , IL17F , IL21 , IL22 and IL26 were comparable"
    - **Explicit Expression**: Any statement or noun phrase indicating that the cytokine is expressed by the cell type, or that the cytokine is a marker for the cell type
        - "TGFB+ Treg cells"
        - "Th2 cells express IL5"
        - "GATA3 is no longer required for **IL-4** expression in differentiated **Th2** cells"
        - "Co-stimulators, such as PD-L2, ICOS-L were higher in CD4+T cells as were the **TFH** markers **CXCR5** and ICOS"
    - **Cytokine Source**: A statement implying that it is possible to obtain the cytokine in question using a particular cell type
        - "Th17 cells or γδ T cells derived IL17 mobilize neutrophils"
    - **Cell Function**: Cytokines may be indicated as a "function" of a cell type or a behavior that the cell type posses
        - "T cell effector functions which were attenuated by TGFβ1-LAP on activating beads included inflammatory responses (TNF-α), TH1 and TH2 functions (IFN-γ, IL-4), TH17 functions (IL-17) and regulatory T cell activities (IL-10)."
- **Negative**
    - **Inducing Cytokine**: Any "Positive" rule for the inducing cytokine relation can be assumed as a disqualifing condition for this relation
    - **Response Composition**: A cytokine that is associated with a how a cell type "responds" to some condition or treatment
        - "Accordingly, the typical M1 biomarkers **IL-1β, IL-6, and IL-12** mRNAs were significantly higher in the untreated TG skin in comparison with the WT skin suggesting the predisposition of the TG skin towards **Th1** responses"
    - **Weak Association**: A relationship implied with no further information, or one based on correlation in an assay
        - "Th2 (IL-13)"
        - "In fact, the ability of non–T cells to secrete IL-10 may limit the differentiation of **IL-10** **T reg** cells"
        - "IL-23 has been primarily linked to the T helper 17 (Th17) cell subset"
        - "These results demonstrated that the expression of **IL-35** was positively correlated with the percent of CD4+FoxP3+ **Tregs**"
        - "Blockade of IL-1R resulted in significantly decreased **Th17** cells in both cultures and decreased **IL-17** levels in culture supernatants"
    - **Proliferation**: A cytokine involved in proliferation or cell type maintenance
        - "creating an environment rich in IL-10 and TGF-β, cytokines known to promote regulatory T cells"
