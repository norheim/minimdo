function deselect(){
            gridOptions.api.deselectAll()
        }
const eGridDiv = document.getElementById("myGrid");
const gridOptions = {
columnDefs: [
  { field: "make",  },
  { field: "model",  },
  { field: "price",  },
],
defaultColDef: {sortable: true, filter: true},
rowData: [
  {
    make: "Vauxhall",
    model: "Corsa",
    price: 17300,
  },
  { make: "Ford", model: "Fiesta", price: 18000 },
  {
    make: "Volkswagen",
    model: "Golf",
    price: 26000,
  },
],
rowSelection:'multiple',
animateRows: true,
onCellClicked: params => {
    console.log('cell was clicked', params)
}
};

new agGrid.Grid(eGridDiv, gridOptions);
fetch(
"https://www.ag-grid.com/example-assets/row-data.json"
)
.then(response => response.json())
.then(data => {
  gridOptions.api.setRowData(data);
});